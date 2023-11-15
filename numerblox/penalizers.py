import scipy
from abc import abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for NumerBlox Penalizers. `pip install tensorflow` first."
    )

class BasePenalizer(BaseEstimator, TransformerMixin):
    """
    Base class for penalization so it is compatible with scikit-learn.
    :param new_col_name: Name of new neutralized column.
    """
    def __init__(self, new_col_name: str):
        self.new_col_name = new_col_name
        super().__init__()

    def fit(self, X=None, y=None, **kwargs):
        return self

    @abstractmethod
    def transform(
        self, X: Union[np.array, pd.DataFrame], 
        features: pd.DataFrame, eras: pd.Series, **kwargs
    ) -> np.array:
        ...

    def predict(self, X: np.array, features: pd.DataFrame, eras: Union[np.array, pd.Series]) -> np.array:
        """ Convenience function for scikit-learn compatibility. """
        return self.transform(X=X, features=features, eras=eras)

    def fit_transform(self, X: np.array, features: pd.DataFrame, eras: Union[np.array, pd.Series]) -> np.array:
        """ 
        Convenience function for scikit-learn compatibility.
        Needed because fit and transform except different arguments here.
        """
        return self.fit().transform(X=X, features=features, eras=eras)
    
    def __call__(
        self, X: Union[np.array, pd.DataFrame],
        features: pd.DataFrame, eras: pd.Series, **kwargs
    ) -> np.array:
        return self.predict(X=X, features=features, eras=eras, **kwargs)
    
    def get_feature_names_out(self, input_features: list = None) -> list:
        """ 
        Get feature names for neutralized output.
        
        :param input_features: Optional list of input feature names.
        :return: List of feature names for neutralized output.
        """
        return input_features if input_features else [self.new_col_name]

class FeaturePenalizer(BasePenalizer):
    """
    Feature penalization with TensorFlow.

    Source (by jrb): https://github.com/jonrtaylor/twitch/blob/master/FE_Clipping_Script.ipynb

    Source of first PyTorch implementation (by Michael Oliver / mdo): https://forum.numer.ai/t/model-diagnostics-feature-exposure/899/12

    :param max_exposure: Number in range [0...1] indicating how much to reduce max feature exposure to.
    :param pred_name: Prediction column name. Used for new column name. \n
    :param suffix: Optional suffix that is added to new column name.
    """
    def __init__(
        self,
        max_exposure: float,
        pred_name: str = "prediction",
        suffix: str = None,
    ):
        self.max_exposure = max_exposure
        self.pred_name = pred_name
        assert (
            0.0 <= max_exposure <= 1.0
        ), f"'max_exposure' should be a float in range [0...1]. Got '{self.max_exposure}'."
        new_col_name = (
            f"{self.pred_name}_penalized_{self.max_exposure}_{suffix}"
            if suffix
            else f"{self.pred_name}_penalized_{self.max_exposure}"
        )
        super().__init__(new_col_name=new_col_name)
        self.suffix = suffix

    def transform(self, X: pd.DataFrame, features: pd.DataFrame, eras: pd.Series) -> np.array:
        """
        Main transform method.
        :param X: Input predictions to neutralize. 
        :param features: DataFrame with features for neutralization. 
        :param eras: Series with era labels for each row in features. 
        Features, eras and the prediction column must all have the same length.
        :return: Penalized predictions.
        """
        assert len(X) == len(features), "Input predictions must have same length as features."
        assert len(X) == len(eras), "Input predictions must have same length as eras."
        df = features.copy()
        df["prediction"] = X
        df["era"] = eras
        penalized_data = self._reduce_all_exposures(
            dataf=df, column=self.pred_name, neutralizers=list(features.columns)
        )
        return penalized_data

    def _reduce_all_exposures(
        self,
        dataf: pd.DataFrame,
        column: str = "prediction",
        neutralizers: list = None,
        normalize=True,
        gaussianize=True,
    ) -> pd.DataFrame:
        neutralized = []

        for era in tqdm(dataf["era"].unique()):
            dataf_era = dataf[dataf["era"] == era]
            scores = dataf_era[[column]].values
            exposure_values = dataf_era[neutralizers].values

            if normalize:
                scores2 = []
                for x in scores.T:
                    x = (scipy.stats.rankdata(x, method="ordinal") - 0.5) / len(x)
                    if gaussianize:
                        x = scipy.stats.norm.ppf(x)
                    scores2.append(x)
                scores = np.array(scores2)[0]

            scores, _ = self._reduce_exposure(
                scores, exposure_values, len(neutralizers), None
            )

            scores /= tf.math.reduce_std(scores)
            scores -= tf.reduce_min(scores)
            scores /= tf.reduce_max(scores)
            neutralized.append(scores.numpy())

        predictions = pd.DataFrame(
            np.concatenate(neutralized), columns=[column], index=dataf.index
        )
        return predictions

    def _reduce_exposure(self, prediction, features, input_size=50, weights=None):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(input_size),
                tf.keras.experimental.LinearModel(use_bias=False),
            ]
        )
        feats = tf.convert_to_tensor(features - 0.5, dtype=tf.float32)
        pred = tf.convert_to_tensor(prediction, dtype=tf.float32)
        if weights is None:
            optimizer = tf.keras.optimizers.Adamax()
            start_exp = self.__exposures(feats, pred[:, None])
            target_exps = tf.clip_by_value(
                start_exp, -self.max_exposure, self.max_exposure
            )
            self._train_loop(model, optimizer, feats, pred, target_exps)
        else:
            model.set_weights(weights)
        return pred[:, None] - model(feats), model.get_weights()

    def _train_loop(self, model, optimizer, feats, pred, target_exps):
        for _ in range(1000000):
            loss, grads = self.__train_loop_body(model, feats, pred, target_exps)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if loss < 1e-7:
                break

    @tf.function(experimental_relax_shapes=True)
    def __train_loop_body(self, model, feats, pred, target_exps):
        with tf.GradientTape() as tape:
            exps = self.__exposures(feats, pred[:, None] - model(feats, training=True))
            loss = tf.reduce_sum(
                tf.nn.relu(tf.nn.relu(exps) - tf.nn.relu(target_exps))
                + tf.nn.relu(tf.nn.relu(-exps) - tf.nn.relu(-target_exps))
            )
        return loss, tape.gradient(loss, model.trainable_variables)

    @staticmethod
    @tf.function(experimental_relax_shapes=True, experimental_compile=True)
    def __exposures(x, y):
        x = x - tf.math.reduce_mean(x, axis=0)
        x = x / tf.norm(x, axis=0)
        y = y - tf.math.reduce_mean(y, axis=0)
        y = y / tf.norm(y, axis=0)
        return tf.matmul(x, y, transpose_a=True)
    
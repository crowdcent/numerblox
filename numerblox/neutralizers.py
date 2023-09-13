
# TODO Inherit from VotingRegressor to create meta estimators.


import scipy
from abc import abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import scipy.stats as sp
from tqdm.auto import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin

from .numerframe import NumerFrame


class BaseNeutralizer(BaseEstimator, RegressorMixin):
    """
    Base class for neutralization so it is compatible with scikit-learn.
    :param new_col_name: Name of new neutralized column.
    """
    def __init__(self, new_col_name: str):
        self.new_col_name = new_col_name
        super().__init__()

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def predict(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> np.array:
        ...
    
    def __call__(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> np.array:
        return self.predict(X=X, y=y, **kwargs)
    
    def get_feature_names_out(self, input_features: list = None) -> list:
        """ 
        Get feature names for neutralized output.
        
        :param input_features: Optional list of input feature names.
        :return: List of feature names for neutralized output.
        """
        return input_features if input_features else [self.new_col_name]


class FeatureNeutralizer(BaseNeutralizer):
    """
    Classic feature neutralization by subtracting a linear model.

    :param feature_names: List of column names to neutralize against. Selects all feature columns by default. \n
    :param pred_name: Prediction column to neutralize. \n
    :param proportion: Number in range [0...1] indicating how much to neutralize. \n
    :param suffix: Optional suffix that is added to new column name. \n
    :param cuda: Do neutralization on the GPU \n
    Make sure you have CuPy installed when setting cuda to True. \n
    Installation docs: docs.cupy.dev/en/stable/install.html
    :param verbose: Whether to print info about neutralization.
    """
    def __init__(
        self,
        feature_names: list = None,
        pred_name: str = "prediction",
        era_col: str = "era",
        proportion: float = 0.5,
        suffix: str = None,
        cuda = False,
        verbose: bool = True,
    ):
        self.pred_name = pred_name
        self.proportion = proportion
        assert (
            0.0 <= self.proportion <= 1.0
        ), f"'proportion' should be a float in range [0...1]. Got '{self.proportion}'."
        new_col_name = (
            f"{self.pred_name}_neutralized_{self.proportion}_{suffix}"
            if suffix
            else f"{self.pred_name}_neutralized_{self.proportion}"
        )
        super().__init__(new_col_name=new_col_name)
        self.era_col = era_col
        self.feature_names = feature_names
        self.cuda = cuda
        self.verbose = verbose

    def predict(self, X: NumerFrame, y=None) -> np.array:
        feature_names = self.feature_names if self.feature_names else X.feature_cols
        neutralized_preds = X.groupby(X.meta.era_col, group_keys=False).apply(
            lambda x: self.normalize_and_neutralize(x, [self.pred_name], feature_names)
        )
        neutralized_preds = MinMaxScaler().fit_transform(
            neutralized_preds
        )
        return neutralized_preds

    def neutralize(self, dataf: pd.DataFrame, columns: list, by: list) -> pd.DataFrame:
        """ Neutralize on CPU. """
        scores = dataf[columns]
        exposures = dataf[by].values
        scores = scores - self.proportion * exposures.dot(
            np.linalg.pinv(exposures).dot(scores)
        )
        return scores / scores.std()

    def neutralize_cuda(self, dataf: pd.DataFrame, columns: list, by: list) -> np.ndarray:
        """ Neutralize on GPU. """
        try:
            import cupy
        except ImportError:
            raise ImportError("CuPy not installed. Set cuda=False or install CuPy. Installation docs: docs.cupy.dev/en/stable/install.html")
        scores = cupy.array(dataf[columns].values)
        exposures = cupy.array(dataf[by].values)
        scores = scores - self.proportion * exposures.dot(
            cupy.linalg.pinv(exposures).dot(scores)
        )
        return cupy.asnumpy(scores / scores.std())

    @staticmethod
    def normalize(dataf: pd.DataFrame) -> np.ndarray:
        normalized_ranks = (dataf.rank(method="first") - 0.5) / len(dataf)
        # Gaussianized
        return sp.norm.ppf(normalized_ranks)

    def normalize_and_neutralize(
        self, dataf: pd.DataFrame, columns: list, by: list
    ) -> pd.DataFrame:
        dataf[columns] = self.normalize(dataf[columns])
        neutralization_func = self.neutralize if not self.cuda else self.neutralize_cuda
        dataf[columns] = neutralization_func(dataf, columns, by)
        return dataf[columns]


class FeaturePenalizer(BaseNeutralizer):
    """
    Feature penalization with TensorFlow.

    Source (by jrb): https://github.com/jonrtaylor/twitch/blob/master/FE_Clipping_Script.ipynb

    Source of first PyTorch implementation (by Michael Oliver / mdo): https://forum.numer.ai/t/model-diagnostics-feature-exposure/899/12

    :param feature_names: List of column names to reduce feature exposure. Uses all feature columns by default. \n
    :param pred_name: Prediction column to neutralize. \n
    :param max_exposure: Number in range [0...1] indicating how much to reduce max feature exposure to.
    """
    def __init__(
        self,
        max_exposure: float,
        feature_names: list = None,
        pred_name: str = "prediction",
        era_col: str = "era",
        suffix: str = None,
    ):
        assert (
            0.0 <= max_exposure <= 1.0
        ), f"'max_exposure' should be a float in range [0...1]. Got '{max_exposure}'."
        new_col_name = (
            f"{self.pred_name}_penalized_{self.max_exposure}_{suffix}"
            if suffix
            else f"{self.pred_name}_penalized_{self.max_exposure}"
        )
        super().__init__(new_col_name=new_col_name)
        self.max_exposure = max_exposure
        self.feature_names = feature_names
        self.pred_name = pred_name
        self.era_col = era_col

    def predict(self, X: pd.DataFrame, y=None) -> np.array:
        penalized_data = self.reduce_all_exposures(
            dataf=X, column=self.pred_name, neutralizers=self.feature_names
        )
        return penalized_data

    def reduce_all_exposures(
        self,
        dataf: NumerFrame,
        column: str = "prediction",
        neutralizers: list = None,
        normalize=True,
        gaussianize=True,
    ) -> pd.DataFrame:
        if neutralizers is None:
            neutralizers = [x for x in dataf.columns if x.startswith("feature")]
        neutralized = []

        for era in tqdm(dataf[self.era_col].unique()):
            dataf_era = dataf[dataf[self.era_col] == era]
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

            scores, weights = self._reduce_exposure(
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
    
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
import scipy.stats as sp
from abc import abstractmethod
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_groups import V4_2_FEATURE_GROUP_MAPPING


class BaseNeutralizer(BaseEstimator, TransformerMixin):
    """
    Base class for neutralization so it is compatible with scikit-learn.
    :param new_col_name: Name of new neutralized column.
    """
    def __init__(self, new_col_names: list):
        self.new_col_names = new_col_names
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
        return input_features if input_features else self.new_col_names


class FeatureNeutralizer(BaseNeutralizer):
    """
    Classic feature neutralization by subtracting a linear model.

    :param pred_name: Name of prediction column. For creating the new column name. \n
    :param proportion: Number in range [0...1] indicating how much to neutralize. \n
    :param suffix: Optional suffix that is added to new column name. \n
    """
    def __init__(
        self,
        pred_name: Union[str, list] = "prediction",
        proportion: Union[float, List[float]] = 0.5,
        suffix: str = None,
    ):
        self.pred_name = [pred_name] if isinstance(pred_name, str) else pred_name
        self.proportion = [proportion] if isinstance(proportion, float) else proportion
        assert len(self.pred_name) == len(set(self.pred_name)), "Duplicate 'pred_names' found. Make sure all names are unique."
        assert len(self.proportion) == len(set(self.proportion)), "Duplicate 'proportions' found. Make sure all proportions are unique."
        for prop in self.proportion:
            assert (
                0.0 <= prop <= 1.0
            ), f"'proportion' should be a float in range [0...1]. Got '{prop}'."

        new_col_names = []
        for pred_name in self.pred_name:
            for prop in self.proportion:
                new_col_names.append(
                    f"{pred_name}_neutralized_{prop}_{suffix}" if suffix else f"{pred_name}_neutralized_{prop}"
                )
        super().__init__(new_col_names=new_col_names)
        self.suffix = suffix

    def transform(self, X: np.array, features: pd.DataFrame, eras: Union[np.array, pd.Series]) -> np.array:
        """
        Main transform function.
        :param X: Input predictions to neutralize. \n
        :param features: DataFrame with features for neutralization. \n
        :param eras: Series with era labels for each row in features. \n
        Features, eras and the prediction column must all have the same length.
        :return: Neutralized predictions.
        """
        assert len(X) == len(features), "Input predictions must have same length as features."
        assert len(X) == len(eras), "Input predictions must have same length as eras."
        df = features.copy()
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for i, pred_name in enumerate(self.pred_name):
            df[pred_name] = X[:, i]
        df["era"] = eras

        feature_cols = list(features.columns)
        tasks = [
            delayed(self._process_pred_name)(df, pred_name, proportion, feature_cols)
            for pred_name in tqdm(self.pred_name, desc="Processing feature neutralizations") 
            for proportion in self.proportion
        ]
        neutralized_results = Parallel(n_jobs=-1)(tasks)
        neutralized_preds = pd.concat(neutralized_results, axis=1).to_numpy()
        return neutralized_preds
    
    def _process_pred_name(self, df, pred_name, proportion, feature_cols):
        neutralized_pred = df.groupby("era", group_keys=False).apply(
            lambda x: self.normalize_and_neutralize(x, [pred_name], feature_cols, proportion)
        )
        return pd.DataFrame(MinMaxScaler().fit_transform(neutralized_pred))

    def neutralize(self, dataf: pd.DataFrame, columns: list, by: list, proportion: float) -> pd.DataFrame:
        """ Neutralize on CPU. """
        scores = dataf[columns]
        exposures = dataf[by].values
        scores = scores - proportion * self._get_raw_exposures(exposures, scores)
        return scores / scores.std()

    @staticmethod
    def normalize(dataf: pd.DataFrame) -> np.ndarray:
        normalized_ranks = (dataf.rank(method="first") - 0.5) / len(dataf)
        # Gaussianized ranks
        return sp.norm.ppf(normalized_ranks)

    def normalize_and_neutralize(
        self, dataf: pd.DataFrame, columns: list, by: list, proportion: float
    ) -> pd.DataFrame:
        dataf[columns] = self.normalize(dataf[columns])
        dataf[columns] = self.neutralize(dataf, columns, by, proportion)
        return dataf[columns]
    
    @staticmethod
    def _get_raw_exposures(exposures: np.array, scores: pd.DataFrame) -> np.array:
        """ 
        Get raw feature exposures.
        Make sure predictions are normalized!
        :param exposures: Exposures for each era. 
        :param scores: DataFrame with predictions.
        :return: Raw exposures for each era.
        """
        return exposures.dot(np.linalg.pinv(exposures).dot(scores))
    
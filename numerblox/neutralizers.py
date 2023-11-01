import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
import scipy.stats as sp
from abc import abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_groups import V4_2_FEATURE_GROUP_MAPPING


class BaseNeutralizer(BaseEstimator, TransformerMixin):
    """
    Base class for neutralization so it is compatible with scikit-learn.
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


class FeatureNeutralizer(BaseNeutralizer):
    """
    Classic feature neutralization by subtracting a linear model.

    :param pred_name: Name of prediction column. For creating the new column name. \n
    :param proportion: Number in range [0...1] indicating how much to neutralize. \n
    :param suffix: Optional suffix that is added to new column name. \n
    :param cuda: Do neutralization on the GPU \n
    Make sure you have CuPy installed when setting cuda to True. \n
    Installation docs: docs.cupy.dev/en/stable/install.html
    """
    def __init__(
        self,
        pred_name: str = "prediction",
        proportion: float = 0.5,
        suffix: str = None,
        cuda = False,
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
        self.suffix = suffix
        self.cuda = cuda

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
        df["prediction"] = X
        df["era"] = eras
        neutralized_preds = df.groupby("era", group_keys=False).apply(
            lambda x: self.normalize_and_neutralize(x, ["prediction"], list(features.columns))
        )
        neutralized_preds = MinMaxScaler().fit_transform(
            neutralized_preds
        )
        return neutralized_preds

    def neutralize(self, dataf: pd.DataFrame, columns: list, by: list) -> pd.DataFrame:
        """ Neutralize on CPU. """
        scores = dataf[columns]
        exposures = dataf[by].values
        scores = scores - self.proportion * self._get_raw_exposures(exposures, scores)
        return scores / scores.std()

    def neutralize_cuda(self, dataf: pd.DataFrame, columns: list, by: list) -> np.ndarray:
        """ Neutralize on GPU. """
        try:
            import cupy
        except ImportError:
            raise ImportError("CuPy not installed. Set cuda=False or install CuPy. Installation docs: docs.cupy.dev/en/stable/install.html")
        scores = cupy.array(dataf[columns].values)
        exposures = cupy.array(dataf[by].values)
        scores = scores - self.proportion * self._get_raw_exposures_cuda(exposures, scores)
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
    
    @staticmethod
    def _get_raw_exposures_cuda(exposures: np.array, scores: pd.DataFrame) -> np.array:
        """ 
        Get raw feature exposures.
        Make sure predictions are normalized!
        :param exposures: Exposures for each era. 
        :param scores: DataFrame with predictions.
        :return: Raw exposures for each era.
        """
        try:
            import cupy
        except ImportError:
            raise ImportError("CuPy not installed. Set cuda=False or install CuPy. Installation docs: docs.cupy.dev/en/stable/install.html")
        return exposures.dot(cupy.linalg.pinv(exposures).dot(scores))
    
    def get_raw_feature_exposures(self, X: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Calculate raw feature exposures for each era.

        :param X: DataFrame containing predictions, features, and eras.
        :param feature_list: List of feature columns in X.
        :return: DataFrame with raw feature exposures by era for each feature.
        """
        # Normalize predictions
        X[f"{self.pred_name}_normalized"] = self.normalize(X[[self.pred_name]])
        
        # Store each feature's exposure data
        feature_exposure_data = pd.DataFrame(index=X["era"].unique(), columns=feature_list)

        for era, group in tqdm(X.groupby("era"), desc="Calculating raw feature exposures"):
            scores = group[f"{self.pred_name}_normalized"].values.reshape(-1, 1)
            exposures = group[feature_list].values
            projected_scores = np.linalg.pinv(exposures).dot(scores)
            feature_exposures_matrix = exposures * projected_scores.T
            feature_exposure_data.loc[era, :] = feature_exposures_matrix.mean(axis=0)
        
        # For ill-conditioned instances, the feature exposure can be very large or small.
        mask = (feature_exposure_data > 1) | (feature_exposure_data < -1)
        feature_exposure_data[mask] = np.nan

        return feature_exposure_data
    
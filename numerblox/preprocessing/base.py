import numpy as np
import pandas as pd
from typing import Union, List
from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class BasePreProcessor(BaseEstimator, TransformerMixin):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        ...

    def fit(self, X, y=None, **kwargs):
        return self

    @abstractmethod
    def transform(
        self, X: Union[np.array, pd.DataFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        ...

    def __call__(
        self, X: Union[np.array, pd.DataFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        return self.transform(X=X, y=y, **kwargs)
    
    @abstractmethod
    def get_feature_names_out(self, input_features=None) -> List[str]:
        ...
    
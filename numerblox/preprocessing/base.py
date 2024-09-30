from abc import abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreProcessor(BaseEstimator, TransformerMixin):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        sklearn.set_config(enable_metadata_routing=True)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    @abstractmethod
    def transform(self, X: Union[np.array, pd.DataFrame], y=None, **kwargs) -> pd.DataFrame: ...

    @abstractmethod
    def get_feature_names_out(self, input_features=None) -> List[str]: ...

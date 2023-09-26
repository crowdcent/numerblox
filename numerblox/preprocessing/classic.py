import warnings
import numpy as np
import pandas as pd
from typing import List

from numerblox.preprocessing.base import BasePreProcessor
from numerblox.feature_groups import V4_2_FEATURE_GROUP_MAPPING

class GroupStatsPreProcessor(BasePreProcessor):
    """
    WARNING: Only supported for v4.2 (Rain) data. The Rain dataset (re)introduced feature groups. \n
    Note that this class only works with `pd.DataFrame` input.
    We using in a Pipeline, make sure that the Pandas output API is set (`.set_output(transform="pandas")`.
    
    Calculates group statistics for all data groups. \n
    :param groups: Groups to create features for. All groups by default. \n
    """
    def __init__(self, groups: list = None):
        super().__init__()
        self.all_groups = [
            'intelligence', 
            'charisma', 
            'strength', 
            'dexterity', 
            'constitution', 
            'wisdom', 
            'agility', 
            'serenity', 
            'sunshine', 
            'rain'
        ]
        self.groups = groups 
        self.group_names = groups if self.groups else self.all_groups
        self.feature_group_mapping = V4_2_FEATURE_GROUP_MAPPING

    def transform(self, X: pd.DataFrame) -> np.array:
        """Check validity and add group features."""
        dataf = self._add_group_features(X)
        return dataf.to_numpy()

    def _add_group_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean, standard deviation and skew for each group."""
        dataf = pd.DataFrame()
        for group in self.group_names:
            cols = self.feature_group_mapping[group]
            valid_cols = [col for col in cols if col in X.columns]
            if not valid_cols:
                warnings.warn(f"None of the columns of '{group}' are in the input data. Output will be nans for the group features.")
            elif len(cols) != len(valid_cols):
                warnings.warn(f"Not all columns of '{group}' are in the input data ({len(valid_cols)} < {len(cols)}). Use remaining columns for stats features.")
            dataf.loc[:, f"feature_{group}_mean"] = X[valid_cols].mean(axis=1)
            dataf.loc[:, f"feature_{group}_std"] = X[valid_cols].std(axis=1)
            dataf.loc[:, f"feature_{group}_skew"] = X[valid_cols].skew(axis=1)
        return dataf
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        if not input_features:
            feature_names = []
            for group in self.group_names:
                feature_names.append(f"feature_{group}_mean")
                feature_names.append(f"feature_{group}_std")
                feature_names.append(f"feature_{group}_skew")
        else:
            feature_names = input_features
        return feature_names
    
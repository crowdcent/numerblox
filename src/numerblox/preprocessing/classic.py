import warnings
from typing import List

import numpy as np
import narwhals as nw
from narwhals.typing import FrameT, IntoFrame

from numerblox.feature_groups import V5_FEATURE_GROUP_MAPPING
from numerblox.preprocessing.base import BasePreProcessor


class GroupStatsPreProcessor(BasePreProcessor):
    """
    Calculates group statistics for all data groups.
    Works with both Pandas and Polars DataFrames.

    :param groups: Groups to create features for. All groups by default.
    """

    def __init__(self, groups: list = None):
        super().__init__()
        self.all_groups = ["intelligence", "charisma", "strength", "dexterity", "constitution", "wisdom", "agility", "serenity", "sunshine", "rain"]
        self.groups = groups
        self.group_names = groups if self.groups else self.all_groups
        self.feature_group_mapping = V5_FEATURE_GROUP_MAPPING

    @nw.narwhalify
    def transform(self, X: IntoFrame) -> np.ndarray:
        """Check validity and add group features."""
        dataf = self._add_group_features(X)
        return dataf.to_numpy()

    @nw.narwhalify
    def _add_group_features(self, X: FrameT) -> FrameT:
        """Mean, standard deviation and skew for each group."""
        result = []
        for group in self.group_names:
            cols = self.feature_group_mapping[group]
            valid_cols = [col for col in cols if col in X.columns]
            if not valid_cols:
                warnings.warn(f"None of the columns of '{group}' are in the input data. Output will be nans for the group features.")
                result.extend([
                    nw.lit(np.nan).alias(f"feature_{group}_mean"),
                    nw.lit(np.nan).alias(f"feature_{group}_std"),
                    # nw.lit(np.nan).alias(f"feature_{group}_skew")
                ])
            else:
                if len(cols) != len(valid_cols):
                    warnings.warn(f"Not all columns of '{group}' are in the input data ({len(valid_cols)} < {len(cols)}). Use remaining columns for stats features.")
                result.extend([
                    nw.col(valid_cols).mean().alias(f"feature_{group}_mean"),
                    nw.col(valid_cols).std().alias(f"feature_{group}_std"),
                    # nw.col(valid_cols).skew().alias(f"feature_{group}_skew")
                ])
        return X.select(result)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        if not input_features:
            feature_names = []
            for group in self.group_names:
                feature_names.extend([
                    f"feature_{group}_mean",
                    f"feature_{group}_std",
                    # f"feature_{group}_skew"
                ])
        else:
            feature_names = input_features
        return feature_names

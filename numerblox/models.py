import pandas as pd
from xgboost import XGBRegressor
from sklearn.utils.validation import check_is_fitted

from .evaluation import NumeraiClassicEvaluator


class EraBoostedXGBRegressor(XGBRegressor):
    """
    Custom XGBRegressor model that upweights the worst eras in the data.
    The worst eras are determined by Corrv2.
    NOTE: Currently only supports single target regression.

    This idea was first proposed by Richard Craib in the Numerai forums:
    https://forum.numer.ai/t/era-boosted-models/189

    Credits to Michael Oliver (mdo) for proposing the 1st XGBoost implementation of era boosting:
    https://forum.numer.ai/t/era-boosted-models/189/3

    :param proportion: Proportion of eras to upweight.
    :param trees_per_step: Number of trees to add per iteration.
    :param num_iters: Number of total era boosting iterations.
    """
    def __init__(self, proportion=0.5, trees_per_step=10, num_iters=200, **xgb_params):
        super().__init__(**xgb_params)
        if not self.n_estimators:
            self.n_estimators = 100
        assert self.n_estimators >= 1, "n_estimators must be at least 1."

        assert 0 < proportion < 1, "proportion must be between 0 and 1."
        self.proportion = proportion
        assert trees_per_step >= 0, "trees_per_step must be at least 1."
        self.trees_per_step = trees_per_step
        assert num_iters >= 2, "num_iters must be at least 2."
        self.num_iters = num_iters

    def fit(self, X, y, eras: pd.Series, **fit_params):
        super().fit(X, y, **fit_params)
        evaluator = NumeraiClassicEvaluator(era_col="era")
        self.feature_names = self.get_booster().feature_names
        iter_df = pd.DataFrame(X, columns=self.feature_names)
        iter_df["target"] = y
        iter_df["era"] = eras

        for _ in range(self.num_iters - 1):
            preds = self.predict(X)
            iter_df["predictions"] = preds
            era_scores = pd.Series(index=iter_df["era"].unique())

            # Per era Corrv2 aka "Numerai Corr".
            era_scores = evaluator.per_era_numerai_corrs(
                dataf=iter_df, pred_col="predictions", target_col="target"
                )
            # Filter on eras with worst Corrv2.
            era_scores.sort_values(inplace=True)
            worst_eras = era_scores[era_scores <= era_scores.quantile(self.proportion)].index
            worst_df = iter_df[iter_df["era"].isin(worst_eras)]

            # Add estimators and fit on worst eras.
            self.n_estimators += self.trees_per_step
            booster = self.get_booster()
            super().fit(worst_df.drop(columns=["target", "era", "predictions"]), 
                        worst_df["target"],
                        xgb_model=booster,
                        **fit_params)
        return self
    
    def get_feature_names_out(self, input_features=None):
        """ Get output feature names for transformation. """
        check_is_fitted(self)
        return self.feature_names if not input_features else input_features
    
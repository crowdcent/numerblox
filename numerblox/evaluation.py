import time
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Union
from numerapi import SignalsAPI
from joblib import Parallel, delayed
from numerai_tools.scoring import correlation_contribution

from .neutralizers import FeatureNeutralizer
from .misc import Key
from .feature_groups import FNCV3_FEATURES

FAST_METRICS = ["mean_std_sharpe", "apy", "max_drawdown", "calmar_ratio"]
ALL_NO_BENCH_METRICS = FAST_METRICS + ["autocorrelation", "max_feature_exposure", "smart_sharpe", "legacy_mean_std_sharpe", "fn_mean_std_sharpe", "tb200_mean_std_sharpe", "tb500_mean_std_sharpe"]
BENCH_METRICS = ["corr_with", "mc_mean_std_sharpe", "legacy_mc_mean_std_sharpe", "ex_diss",
                 "ex_diss_pearson", "ex_diss_spearman"]
CLASSIC_SPECIFIC_METRICS = ["fncv3_mean_std_sharpe"]
ALL_COMMON_METRICS = FAST_METRICS + ALL_NO_BENCH_METRICS + BENCH_METRICS


MINI_METRICS = ["mean_std_sharpe"]
CORE_METRICS = FAST_METRICS + ["max_feature_exposure", "smart_sharpe", "legacy_mean_std_sharpe"] + BENCH_METRICS
ALL_CLASSIC_METRICS = ALL_COMMON_METRICS + CLASSIC_SPECIFIC_METRICS
ALL_SIGNALS_METRICS = ALL_COMMON_METRICS


class BaseEvaluator:
    """
    Evaluation functionality that is relevant for both
    Numerai Classic and Numerai Signals.

    Metrics include:
    - Mean, Standard Deviation and Sharpe (Corrv2) for era returns.
    - Max drawdown.
    - Annual Percentage Yield (APY).
    - Correlation with benchmark predictions.
    - Max feature exposure: https://forum.numer.ai/t/model-diagnostics-feature-exposure/899.
    - Feature Neutral Mean, Standard deviation and Sharpe: https://docs.numer.ai/tournament/feature-neutral-correlation.
    - Smart Sharpe
    - Exposure Dissimilarity: https://forum.numer.ai/t/true-contribution-details/5128/4.
    - Autocorrelation (1st order).
    - Calmar Ratio.
    - Performance vs. Benchmark predictions.
    - Mean, Standard Deviation and Sharpe for TB200 (Buy top 200 stocks and sell bottom 200 stocks).
    - Mean, Standard Deviation and Sharpe for TB500 (Buy top 500 stocks and sell bottom 500 stocks).

    :param metrics_list: List of metrics to calculate. Default: FAST_METRICS.
    :param era_col: Column name pointing to eras. Most commonly "era" for Numerai Classic and "friday_date" for Numerai Signals.
    :param custom_functions: Additional functions called in evaluation.
    Check out the NumerBlox docs on evaluation for more info on using custom functions.
    :param show_detailed_progress_bar: Show detailed progress bar for evaluation of each prediction column.

    Note that we calculate the sample standard deviation with ddof=0.
    It may differ slightly from the standard Pandas calculation, but
    is consistent with how NumPy computes standard deviation.
    More info:
    https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
    """

    def __init__(
        self,
        metrics_list: List[str],
        era_col: str,
        custom_functions: Dict[str, Dict[str, Any]],
        show_detailed_progress_bar: bool,
    ):
        self.era_col = era_col
        self.metrics_list = metrics_list
        self.custom_functions = custom_functions
        if self.custom_functions is not None:
            self.check_custom_functions()
        self.show_detailed_progress_bar = show_detailed_progress_bar

    def full_evaluation(
        self,
        dataf: pd.DataFrame,
        pred_cols: List[str],
        target_col: str = "target",
        benchmark_cols: list = None,
    ) -> pd.DataFrame:
        """
        Perform evaluation for each prediction column in pred_cols.
        By default only the "prediction" column is evaluated.
        Evaluation is done against given target and benchmark prediction column.
        :param dataf: DataFrame containing era_col, pred_cols, target_col and optional benchmark_cols.
        :param pred_cols: List of prediction columns to calculate evaluation metrics for.
        :param target_col: Target column to evaluate against.
        :param benchmark_cols: Optional list of benchmark columns to calculate evaluation metrics for.
        """
        val_stats = pd.DataFrame()
        feature_cols = [col for col in dataf.columns if col.startswith("feature")]
        cat_cols = (
            dataf[feature_cols].select_dtypes(include=["category"]).columns.to_list()
        )
        if cat_cols:
            print(
                f"WARNING: Categorical features detected that cannot be used for neutralization. Removing columns: '{cat_cols}' for evaluation."
            )
            dataf.loc[:, feature_cols] = dataf[feature_cols].select_dtypes(
                exclude=["category"]
            )
        dataf = dataf.fillna(0.5)
        for col in tqdm(pred_cols, desc="Evaluation: "):
            col_stats = self.evaluation_one_col(
                dataf=dataf,
                pred_col=col,
                feature_cols=feature_cols,
                target_col=target_col,
                benchmark_cols=benchmark_cols,
            )
            val_stats = pd.concat([val_stats, col_stats], axis=0)
        return val_stats

    def evaluation_one_col(
        self,
        dataf: pd.DataFrame,
        feature_cols: list,
        pred_col: str,
        target_col: str,
        benchmark_cols: list = None,
    ):
        """
        Perform evaluation for one prediction column
        against given target and benchmark column(s).
        """
        assert (
            self.era_col in dataf.columns
        ), f"Era column '{self.era_col}' not found in DataFrame. Make sure to set the correct era_col."
        assert (
                pred_col in dataf.columns
            ), f"Prediction column '{pred_col}' not found in DataFrame. Make sure to set the correct pred_col."
        assert (
            target_col in dataf.columns
        ), f"Target column '{target_col}' not found in DataFrame. Make sure to set the correct target_col."
        if benchmark_cols:
            for col in benchmark_cols:
                assert (
                    col in dataf.columns
                ), f"Benchmark column '{col}' not found in DataFrame. Make sure to set the correct benchmark_cols."

        # Check that all values are between 0 and 1
        assert (
            dataf[pred_col].min().min() >= 0 and dataf[pred_col].max().max() <= 1
        ), "All predictions should be between 0 and 1 (inclusive)."
        assert (
            dataf[target_col].min() >= 0 and dataf[target_col].max() <= 1
        ), "All targets should be between 0 and 1 (inclusive)."
        if benchmark_cols is not None:
            for col in benchmark_cols:
                assert (
                    dataf[col].min() >= 0 and dataf[col].max() <= 1
                ), f"All predictions for '{col}' should be between 0 and 1 (inclusive)."

        if self.show_detailed_progress_bar:
            len_metrics_list = len(self.metrics_list)
            len_benchmark_cols = 0 if benchmark_cols is None else len(benchmark_cols)
            len_custom_functions = 0 if self.custom_functions is None else len(list(self.custom_functions.keys()))
            len_pbar = len_metrics_list + len_benchmark_cols + len_custom_functions
            pbar = tqdm(total=len_pbar, desc="Evaluation")

        col_stats = {}
        col_stats["target"] = target_col

        # Compute stats per era (only if needed)
        per_era_numerai_corrs = self.per_era_numerai_corrs(
            dataf=dataf, pred_col=pred_col, target_col=target_col
        )

        # check if mean, std, or sharpe are in metrics_list
        if "mean_std_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"mean_std_sharpe for evaluation")
                pbar.update(1)
            mean, std, sharpe = self.mean_std_sharpe(era_corrs=per_era_numerai_corrs)
            col_stats["mean"] = mean
            col_stats["std"] = std
            col_stats["sharpe"] = sharpe

        if "legacy_mean_std_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"legacy_mean_std_sharpe for evaluation")
                pbar.update(1)
            per_era_corrs = self.per_era_corrs(
                dataf=dataf, pred_col=pred_col, target_col=target_col
            )
            legacy_mean, legacy_std, legacy_sharpe = self.mean_std_sharpe(
                era_corrs=per_era_corrs
            )
            col_stats["legacy_mean"] = legacy_mean
            col_stats["legacy_std"] = legacy_std
            col_stats["legacy_sharpe"] = legacy_sharpe

        if "max_drawdown" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"max_drawdown for evaluation")
                pbar.update(1)
            col_stats["max_drawdown"] = self.max_drawdown(
                era_corrs=per_era_numerai_corrs
            )

        if "apy":
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"apy for evaluation")
                pbar.update(1)
            col_stats["apy"] = self.apy(era_corrs=per_era_numerai_corrs)

        if "calmar_ratio" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"calmar_ratio for evaluation")
                pbar.update(1)
            if not "max_drawdown" in self.metrics_list:
                col_stats["max_drawdown"] = self.max_drawdown(
                    era_corrs=per_era_numerai_corrs
                )
            if not "apy" in self.metrics_list:
                col_stats["apy"] = self.apy(era_corrs=per_era_numerai_corrs)
            col_stats["calmar_ratio"] = (
                np.nan
                if col_stats["max_drawdown"] == 0
                else col_stats["apy"] / -col_stats["max_drawdown"]
            )

        if "autocorrelation" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description(f"autocorrelation for evaluation")
                pbar.update(1)
            col_stats["autocorrelation"] = self.autocorr1(per_era_numerai_corrs)

        if "max_feature_exposure" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"max_feature_exposure for evaluation")
                pbar.update(1)
            col_stats["max_feature_exposure"] = self.max_feature_exposure(
                dataf=dataf, feature_cols=feature_cols, pred_col=pred_col
            )

        if "smart_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"smart_sharpe for evaluation")
                pbar.update(1)
            col_stats["smart_sharpe"] = self.smart_sharpe(
                era_corrs=per_era_numerai_corrs
            )

        if benchmark_cols is not None:
            for bench_col in benchmark_cols:
                if self.show_detailed_progress_bar:
                    pbar.set_description_str(f"Evaluation for benchmark column: '{bench_col}'")
                    pbar.update(1)

                per_era_bench_corrs = self.per_era_numerai_corrs(
                    dataf=dataf, pred_col=bench_col, target_col=target_col
                )

                if "mean_std_sharpe" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"mean_std_sharpe for benchmark column: '{bench_col}'")
                    bench_mean, bench_std, bench_sharpe = self.mean_std_sharpe(
                        era_corrs=per_era_bench_corrs
                    )
                    col_stats[f"mean_vs_{bench_col}"] = mean - bench_mean
                    col_stats[f"std_vs_{bench_col}"] = std - bench_std
                    col_stats[f"sharpe_vs_{bench_col}"] = sharpe - bench_sharpe

                if "mc_mean_std_sharpe" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"mc_mean_std_sharpe for benchmark column: '{bench_col}'")
                    mc_scores = self.contributive_correlation(
                        dataf=dataf,
                        pred_col=pred_col,
                        target_col=target_col,
                        other_col=bench_col,
                    )
                    col_stats[f"mc_mean_{bench_col}"] = np.nanmean(mc_scores)
                    col_stats[f"mc_std_{bench_col}"] = np.nanstd(mc_scores)
                    col_stats[f"mc_sharpe_{bench_col}"] = (
                        np.nan
                        if col_stats[f"mc_std_{bench_col}"] == 0
                        else col_stats[f"mc_mean_{bench_col}"]
                        / col_stats[f"mc_std_{bench_col}"]
                    )

                if "corr_with" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"corr_with for benchmark column: '{bench_col}'")
                    col_stats[f"corr_with_{bench_col}"] = self.cross_correlation(
                        dataf=dataf, pred_col=bench_col, other_col=bench_col
                    )

                if "legacy_mc_mean_std_sharpe" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"legacy_mc_mean_std_sharpe for benchmark column: '{bench_col}'")
                    legacy_mc_scores = self.legacy_contribution(
                        dataf=dataf,
                        pred_col=pred_col,
                        target_col=target_col,
                        other_col=bench_col,
                    )
                    col_stats[f"legacy_mc_mean_{bench_col}"] = np.nanmean(
                        legacy_mc_scores
                    )
                    col_stats[f"legacy_mc_std_{bench_col}"] = np.nanstd(
                        legacy_mc_scores
                    )
                    col_stats[f"legacy_mc_sharpe_{bench_col}"] = (
                        np.nan
                        if col_stats[f"legacy_mc_std_{bench_col}"] == 0
                        else col_stats[f"legacy_mc_mean_{bench_col}"]
                        / col_stats[f"legacy_mc_std_{bench_col}"]
                    )

                if "ex_diss" in self.metrics_list or "ex_diss_pearson" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"ex_diss_pearson for benchmark column: '{bench_col}'")
                    col_stats[
                        f"exposure_dissimilarity_pearson_{bench_col}"
                    ] = self.exposure_dissimilarity(
                        dataf=dataf, pred_col=pred_col, other_col=bench_col,
                        corr_method="pearson"
                    )
                if "ex_diss_spearman" in self.metrics_list:
                    if self.show_detailed_progress_bar:
                        pbar.set_description_str(f"ex_diss_spearman for benchmark column: '{bench_col}'")
                    col_stats[
                        f"exposure_dissimilarity_spearman_{bench_col}"
                    ] = self.exposure_dissimilarity(
                        dataf=dataf, pred_col=pred_col, other_col=bench_col,
                        corr_method="spearman"
                    )

        # Compute intensive stats
        if "fn_mean_std_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"fn_mean_std_sharpe for evaluation")
                pbar.update(1)
            fn_mean, fn_std, fn_sharpe = self.feature_neutral_mean_std_sharpe(
                dataf=dataf,
                pred_col=pred_col,
                target_col=target_col,
                feature_names=feature_cols,
            )
            col_stats["feature_neutral_mean"] = fn_mean
            col_stats["feature_neutral_std"] = fn_std
            col_stats["feature_neutral_sharpe"] = fn_sharpe

        if "tb200_mean_std_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"tb200_mean_std_sharpe for evaluation")
                pbar.update(1)
            tb200_mean, tb200_std, tb200_sharpe = self.tbx_mean_std_sharpe(
                dataf=dataf, pred_col=pred_col, target_col=target_col, tb=200
            )
            col_stats["tb200_mean"] = tb200_mean
            col_stats["tb200_std"] = tb200_std
            col_stats["tb200_sharpe"] = tb200_sharpe

        if "tb500_mean_std_sharpe" in self.metrics_list:
            if self.show_detailed_progress_bar:
                pbar.set_description_str(f"tb500_mean_std_sharpe for evaluation")
                pbar.update(1)
            tb500_mean, tb500_std, tb500_sharpe = self.tbx_mean_std_sharpe(
                dataf=dataf, pred_col=pred_col, target_col=target_col, tb=500
            )
            col_stats["tb500_mean"] = tb500_mean
            col_stats["tb500_std"] = tb500_std
            col_stats["tb500_sharpe"] = tb500_sharpe

        # Custom functions
        if self.custom_functions is not None:
            local_vars = locals()
            for func_name, func_info in self.custom_functions.items():
                if self.show_detailed_progress_bar:
                    pbar.set_description_str(f"custom function: '{func_name}' for evaluation")
                    pbar.update(1)
                func = func_info['func']
                args = func_info['args']
                local_args = func_info['local_args']
                resolved_args = {}
                for k, v in args.items():
                    # Resolve variables defined as local args
                    if isinstance(v, str) and v in local_args:
                        if v not in local_vars:
                            raise ValueError(f"Variable '{v}' was defined in 'local_args', but was not found in local variables. Make sure to set the correct local_args.")
                        else:
                            resolved_args[k] = local_vars[v]
                    else:
                        resolved_args[k] = v
                col_stats[func_name] = func(**resolved_args)

        col_stats_df = pd.DataFrame(col_stats, index=[pred_col])
        if self.show_detailed_progress_bar:
            pbar.update(1)
            pbar.close()
        return col_stats_df

    def per_era_corrs(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str
    ) -> pd.Series:
        """Correlation between prediction and target for each era."""
        return dataf.groupby(self.era_col).apply(
            lambda d: self._normalize_uniform(d[pred_col].fillna(0.5)).corr(
                d[target_col]
            )
        )

    def per_era_numerai_corrs(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str
    ) -> pd.Series:
        """Numerai Corr between prediction and target for each era."""
        return dataf.groupby(self.era_col).apply(
            lambda d: self.numerai_corr(d.fillna(0.5), pred_col, target_col)
        )

    def mean_std_sharpe(
        self, era_corrs: pd.Series
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Average, standard deviation and Sharpe ratio for
        correlations per era.
        """
        mean = pd.Series(era_corrs.mean()).item()
        std = pd.Series(era_corrs.std(ddof=0)).item()
        sharpe = np.nan if std == 0 else mean / std
        return mean, std, sharpe

    def numerai_corr(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str
    ) -> np.float64:
        """
        Computes 'Numerai Corr' aka 'Corrv2'.
        More info: https://forum.numer.ai/t/target-cyrus-new-primary-target/6303

        Assumes original target col as input (i.e. in [0, 1] range).
        """
        # Rank and gaussianize predictions
        ranked_preds = self._normalize_uniform(
            dataf[pred_col].fillna(0.5), method="average"
        )
        gauss_ranked_preds = stats.norm.ppf(ranked_preds)
        # Center target from [0...1] to [-0.5...0.5] range
        targets = dataf[target_col]
        centered_target = targets - targets.mean()
        # Accentuate tails of predictions and targets
        preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
        target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
        # Pearson correlation
        corr, _ = stats.pearsonr(preds_p15, target_p15)
        return corr

    @staticmethod
    def max_drawdown(era_corrs: pd.Series) -> np.float64:
        """Maximum drawdown per era."""
        # Arbitrarily large window
        rolling_max = (
            (era_corrs + 1).cumprod().rolling(window=9000, min_periods=1).max()
        )
        daily_value = (era_corrs + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        return max_drawdown

    @staticmethod
    def apy(era_corrs: pd.Series, stake_compounding_lag: int = 4) -> np.float64:
        """
        Annual percentage yield.
        :param era_corrs: Correlation scores by era
        :param stake_compounding_lag: Compounding lag for Numerai rounds (4 for Numerai Classic)
        """
        payout_scores = era_corrs.clip(-0.25, 0.25)
        payout_product = (payout_scores + 1).prod()
        return (
            payout_product
            ** (
                # 52 weeks of compounding minus n for stake compounding lag
                (52 - stake_compounding_lag)
                / len(payout_scores)
            )
            - 1
        ) * 100

    def cross_correlation(self, dataf: pd.DataFrame, pred_col: str, other_col: str):
        """
        Corrv2 correlation with other predictions (like another model, example predictions or meta model prediction).
        :param dataf: DataFrame containing both pred_col and other_col.
        :param pred_col: Main Prediction.
        :param other_col: Other prediction column to calculate correlation with pred_col.

        :return: Correlation between Corrv2's of pred_col and other_col.
        """
        return self.per_era_numerai_corrs(
            dataf=dataf,
            pred_col=pred_col,
            target_col=other_col,
        ).mean()

    def max_feature_exposure(
        self, dataf: pd.DataFrame, feature_cols: List[str], pred_col: str
    ) -> np.float64:
        """Maximum exposure over all features."""
        max_per_era = dataf.groupby(self.era_col).apply(
            lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max()
        )
        max_feature_exposure = max_per_era.mean(skipna=True)
        return max_feature_exposure

    def feature_neutral_mean_std_sharpe(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str, feature_names: list
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Feature neutralized mean performance.
        More info: https://docs.numer.ai/tournament/feature-neutral-correlation
        """
        fn = FeatureNeutralizer(pred_name=pred_col, proportion=1.0)
        neutralized_preds = fn.predict(
            dataf[pred_col], features=dataf[feature_names], eras=dataf[self.era_col]
        )
        # Construct new DataFrame with era col, target col and preds
        neutralized_dataf = pd.DataFrame(columns=[self.era_col, target_col, pred_col])
        neutralized_dataf[self.era_col] = dataf[self.era_col]
        neutralized_dataf[target_col] = dataf[target_col]
        neutralized_dataf[pred_col] = neutralized_preds

        neutral_corrs = self.per_era_numerai_corrs(
            dataf=neutralized_dataf,
            pred_col=pred_col,
            target_col=target_col,
        )
        mean, std, sharpe = self.mean_std_sharpe(era_corrs=neutral_corrs)
        return mean, std, sharpe

    def tbx_mean_std_sharpe(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str, tb: int = 200
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Calculate Mean, Standard deviation and Sharpe ratio
        when we focus on the x top and x bottom predictions.
        :param tb: How many of top and bottom predictions to focus on.
        TB200 and TB500 are the most common situations.
        """
        tb_val_corrs = self._score_by_date(
            dataf=dataf, columns=[pred_col], target=target_col, tb=tb
        )
        return self.mean_std_sharpe(era_corrs=tb_val_corrs)

    def exposure_dissimilarity(
        self, dataf: pd.DataFrame, pred_col: str, other_col: str, corr_method: str = "pearson"
    ) -> np.float32:
        """
        Model pattern of feature exposure to the another column.
        See TC details forum post: https://forum.numer.ai/t/true-contribution-details/5128/4
        :param dataf: DataFrame containing both pred_col and other_col.
        :param pred_col: Main Prediction.
        :param other_col: Other prediction column to calculate exposure dissimilarity against.
        :param corr_method: Correlation method to use for calculating feature exposures.
        corr_method should be one of ['pearson', 'kendall', 'spearman']. Default: 'pearson'.
        """
        assert corr_method in ["pearson", "kendall", "spearman"], f"corr_method should be one of ['pearson', 'kendall', 'spearman']. Got: '{corr_method}'"
        feature_cols = [col for col in dataf.columns if col.startswith("feature")]
        U = dataf[feature_cols].corrwith(dataf[pred_col], method=corr_method).values
        E = dataf[feature_cols].corrwith(dataf[other_col], method=corr_method).values

        denominator = np.dot(E, E)
        if denominator == 0:
            exp_dis = 0
        else:
            exp_dis = 1 - np.dot(U, E) / denominator
        return exp_dis

    @staticmethod
    def _neutralize_series(
        series: pd.Series, by: pd.Series, proportion=1.0
    ) -> pd.Series:
        scores = series.values.reshape(-1, 1)
        exposures = by.values.reshape(-1, 1)

        # This line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
        exposures = np.hstack(
            (exposures, np.array([np.nanmean(series)] * len(exposures)).reshape(-1, 1))
        )

        correction = proportion * (
            exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
        )
        corrected_scores = scores - correction
        neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
        return neutralized

    @staticmethod
    def _orthogonalize(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Orthogonalizes v with respect to u by projecting v onto u,
        then subtracting that projection from v.

        This will reach the same result as the neutralize function when v and u
        are single column vectors, but this is much faster.

        Arguments:
            v: np.ndarray - the vector to orthogonalize
            u: np.ndarray - the vector to orthogonalize v against

        Returns:
            np.ndarray - the orthogonalized vector v
        """
        # Calculate the dot product of u and v
        dot_product = u.T @ v

        # Calculate the projection of v onto u
        projection = (dot_product / (u.T @ u)) * u

        # Subtract the projection from v
        return v - projection

    def _score_by_date(
        self, dataf: pd.DataFrame, columns: list, target: str, tb: int = None
    ):
        """
        Get era correlation based on given TB (x top and bottom predictions).
        :param tb: How many of top and bottom predictions to focus on.
        TB200 is the most common situation.
        """
        unique_eras = dataf[self.era_col].unique()
        computed = []
        for u in unique_eras:
            df_era = dataf[dataf[self.era_col] == u]
            era_pred = np.float64(df_era[columns].values.T)
            era_target = np.float64(df_era[target].values.T)

            if tb is None:
                ccs = np.corrcoef(era_target, era_pred)[0, 1:]
            else:
                tbidx = np.argsort(era_pred, axis=1)
                tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
                ccs = [
                    np.corrcoef(era_target[idx], pred[idx])[0, 1]
                    for idx, pred in zip(tbidx, era_pred)
                ]
                ccs = np.array(ccs)
            computed.append(ccs)
        return pd.DataFrame(
            np.array(computed), columns=columns, index=dataf[self.era_col].unique()
        )

    @staticmethod
    def _normalize_uniform(df: pd.DataFrame, method: str = "first") -> pd.Series:
        """
        Normalize predictions uniformly using ranks.
        NOTE: Make sure the range of predictions is [0, 1] (inclusive).
        """
        x = (df.rank(method=method) - 0.5) / len(
            df
        )  # TODO: Evaluate if subtracting df.mean() is better
        return pd.Series(x, index=df.index)

    def get_feature_exposures_pearson(
        self,
        dataf: pd.DataFrame,
        pred_col: str,
        feature_list: List[str],
        cpu_cores: int = -1,
    ) -> pd.DataFrame:
        """
        Calculate feature exposures for each era using Pearson correlation.

        :param dataf: DataFrame containing predictions, features, and eras.
        :param pred_col: Prediction column to calculate feature exposures for.
        :param feature_list: List of feature columns in X.
        :param cpu_cores: Number of CPU cores to use for parallelization.
        :return: DataFrame with Pearson feature exposures by era for each feature.
        """

        def calculate_era_pearson_exposure(
            era, group, feature_list, pred_col_normalized
        ):
            data_matrix = group[feature_list + [pred_col_normalized]].values
            correlations = np.corrcoef(data_matrix, rowvar=False)

            # Get the correlations of all features with the predictions (which is the last column)
            feature_correlations = correlations[:-1, -1]
            return era, feature_correlations

        normalized_ranks = (dataf[[pred_col]].rank(method="first") - 0.5) / len(dataf)
        dataf[f"{pred_col}_normalized"] = stats.norm.ppf(normalized_ranks)
        feature_exposure_data = pd.DataFrame(
            index=dataf["era"].unique(), columns=feature_list
        )

        grouped_data = list(dataf.groupby("era"))

        results = Parallel(n_jobs=cpu_cores)(
            delayed(calculate_era_pearson_exposure)(
                era, group, feature_list, f"{pred_col}_normalized"
            )
            for era, group in grouped_data
        )

        for era, feature_correlations in results:
            feature_exposure_data.loc[era, :] = feature_correlations
        return feature_exposure_data

    def get_feature_exposures_corrv2(
        self,
        dataf: pd.DataFrame,
        pred_col: str,
        feature_list: List[str],
        cpu_cores: int = -1,
    ) -> pd.DataFrame:
        """
        Calculate feature exposures for each era using 'Numerai Corr'.
        Results will be similar to get_feature_exposures() but more accurate.
        This method will take longer to compute.

        :param dataf: DataFrame containing predictions, features, and eras.
        :param pred_col: Prediction column to calculate feature exposures for.
        :param feature_list: List of feature columns in X.
        :param cpu_cores: Number of CPU cores to use for parallelization.
        Default: -1 (all cores).
        :return: DataFrame with Corrv2 feature exposures by era for each feature.
        """

        def calculate_era_feature_exposure(era, group, pred_col, feature_list):
            exposures = {}
            for feature in feature_list:
                corr = self.numerai_corr(
                    group, pred_col=f"{pred_col}_normalized", target_col=feature
                )
                exposures[feature] = corr
            return era, exposures

        normalized_ranks = (dataf[[pred_col]].rank(method="first") - 0.5) / len(dataf)
        dataf[f"{pred_col}_normalized"] = stats.norm.ppf(normalized_ranks)
        feature_exposure_data = pd.DataFrame(
            index=dataf["era"].unique(), columns=feature_list
        )

        grouped_data = list(dataf.groupby("era"))

        results = Parallel(n_jobs=cpu_cores)(
            delayed(calculate_era_feature_exposure)(era, group, pred_col, feature_list)
            for era, group in grouped_data
        )
        for era, exposures in results:
            feature_exposure_data.loc[era, :] = exposures
        return feature_exposure_data

    def smart_sharpe(self, era_corrs: pd.Series) -> np.float64:
        """
        Sharpe adjusted for autocorrelation.
        :param era_corrs: Correlation scores by era
        """
        return np.nanmean(era_corrs) / (
            np.nanstd(era_corrs, ddof=1) * self.autocorr_penalty(era_corrs)
        )

    def autocorr_penalty(self, era_corrs: pd.Series) -> np.float64:
        """
        Adjusting factor for autocorrelation. Used in Smart Sharpe.
        :param era_corrs: Correlation scores by era.
        """
        n = len(era_corrs)
        # 1st order autocorrelation
        p = self.autocorr1(era_corrs)
        return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p**i for i in range(1, n)]))

    def autocorr1(self, era_corrs: pd.Series) -> np.float64:
        """
        1st order autocorrelation.
        :param era_corrs: Correlation scores by era.
        """
        return np.corrcoef(era_corrs[:-1], era_corrs[1:])[0, 1]

    def legacy_contribution(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str, other_col: str
    ):
        """
        Legacy contibution mean, standard deviation and sharpe ratio.
        More info: https://forum.numer.ai/t/mmc2-announcement/93

        :param dataf: DataFrame containing era_col, pred_col, target_col and other_col.
        :param pred_col: Prediction column to calculate MMC for.
        :param target_col: Target column to calculate MMC against.
        :param other_col: Meta model column containing predictions to neutralize against.

        :return: List of legacy contribution scores by era.
        """
        legacy_mc_scores = []
        # Standard deviation of a uniform distribution
        COVARIANCE_FACTOR = 0.29**2
        # Calculate MMC for each era
        for _, x in dataf.groupby(self.era_col):
            series = self._neutralize_series(
                self._normalize_uniform(x[pred_col]), (x[other_col])
            )
            legacy_mc_scores.append(
                np.cov(series, x[target_col])[0, 1] / COVARIANCE_FACTOR
            )

        return legacy_mc_scores

    def contributive_correlation(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str, other_col: str
    ) -> np.array:
        """Calculate the contributive correlation of the given predictions
        wrt the given meta model.
        see: https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc-and-bmc

        Uses Numerai's official scoring function for contribution under the hood.
        See: https://github.com/numerai/numerai-tools/blob/master/numerai_tools/scoring.py
        
        Calculate contributive correlation by:
        1. tie-kept ranking each prediction and the meta model
        2. gaussianizing each prediction and the meta model
        3. orthogonalizing each prediction wrt the meta model
        3.5. scaling the targets to buckets [-2, -1, 0, 1, 2]
        4. dot product the orthogonalized predictions and the targets
       then normalize by the length of the target (equivalent to covariance)

        :param dataf: DataFrame containing era_col, pred_col, target_col and other_col.
        :param pred_col: Prediction column to calculate MMC for.
        :param target_col: Target column to calculate MMC against.
        Make sure the range of targets is [0, 1] (inclusive). 
        If the function is called from full_evalation, this is guaranteed because of the checks.
        :param other_col: Meta model column containing predictions to neutralize against.

        :return: A 1D NumPy array of contributive correlations by era.
        """
        mc_scores = []
        for _, x in dataf.groupby(self.era_col):
            mc = correlation_contribution(x[[pred_col]], 
                                          x[other_col], 
                                          x[target_col])
            mc_scores.append(mc)
        return np.array(mc_scores).ravel()

    def check_custom_functions(self):
        if not isinstance(self.custom_functions, dict):
            raise ValueError("custom_functions must be a dictionary")

        for func_name, func_info in self.custom_functions.items():
            if not isinstance(func_info, dict) or 'func' not in func_info or 'args' not in func_info:
                raise ValueError(f"Function {func_name} must have a 'func' and 'args' key")

            if not callable(func_info['func']):
                raise ValueError(f"The 'func' value for '{func_name}' in custom_functions must be a callable function.")

            if not isinstance(func_info['args'], dict):
                raise ValueError(f"'args' for '{func_name}' in custom_functions must be a dictionary")
            
            if "local_args" in func_info:
                if not isinstance(func_info['local_args'], list):
                    raise ValueError(f"The 'local_args' key for {func_name} in custom_functionsmust be a list")
                for local_arg in func_info['local_args']:
                    if not isinstance(local_arg, str):
                        raise ValueError(f"Local arg '{local_arg}' for '{func_name}' in custom_functions must be string.")
                    if local_arg not in list(func_info['args'].keys()):
                        raise ValueError(f"Local arg '{local_arg}' for '{func_name}' in custom_functions was not found in 'args'")

    def plot_correlations(
        self,
        dataf: pd.DataFrame,
        pred_cols: List[str],
        corr_cols: list = None,
        target_col: str = "target",
        roll_mean: int = 20,
    ):
        """
        Plot per era correlations over time.
        :param dataf: DataFrame that contains at least all pred_cols, target_col and corr_cols.
        :param pred_cols: List of prediction columns to calculate per era correlations for and plot.
        :param corr_cols: Per era correlations already prepared to include in the plot.
        This is optional for if you already have per era correlations prepared in your input dataf.
        :param target_col: Target column name to compute per era correlations against.
        :param roll_mean: How many eras should be averaged to compute a rolling score.
        """
        validation_by_eras = pd.DataFrame()
        # Compute per era correlation for each prediction column.
        for pred_col in pred_cols:
            per_era_corrs = self.per_era_numerai_corrs(
                dataf, pred_col=pred_col, target_col=target_col
            )
            validation_by_eras.loc[:, pred_col] = per_era_corrs

        # Add prepared per era correlation if any.
        if corr_cols is not None:
            for corr_col in corr_cols:
                validation_by_eras.loc[:, corr_col] = dataf[corr_col]

        validation_by_eras.rolling(roll_mean).mean().plot(
            kind="line",
            marker="o",
            ms=4,
            title=f"Rolling Per Era Correlation Mean (rolling window size: {roll_mean})",
            figsize=(15, 5),
        )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        plt.axhline(y=0.0, color="r", linestyle="--")
        plt.show()

        validation_by_eras.cumsum().plot(
            title="Cumulative Sum of Era Correlations", figsize=(15, 5)
        )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        plt.axhline(y=0.0, color="r", linestyle="--")
        plt.show()
        return

    @staticmethod
    def plot_correlation_heatmap(dataf: pd.DataFrame, pred_cols: List[str]):
        corr_matrix = dataf[pred_cols].corr().to_numpy()

        plt.figure(figsize=(20, 20))

        # Create heatmap
        plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
        plt.colorbar()

        # Add ticks and labels
        ticks = np.arange(0, len(pred_cols), 1)
        plt.xticks(ticks, pred_cols, rotation=90, fontsize=8)
        plt.yticks(ticks, pred_cols, fontsize=8)

        plt.show()
        return


class NumeraiClassicEvaluator(BaseEvaluator):
    """
    Evaluator for all metrics that are relevant in Numerai Classic.
    """

    def __init__(
        self,
        era_col: str = "era",
        metrics_list: List[str] = FAST_METRICS,
        custom_functions: Dict[str, Dict[str, Any]] = None,
        show_detailed_progress_bar: bool = True,
    ):
        for metric in metrics_list:
            assert (
                metric in ALL_CLASSIC_METRICS
            ), f"Metric '{metric}' not found. Valid metrics: {ALL_CLASSIC_METRICS}."
        super().__init__(
            era_col=era_col, metrics_list=metrics_list, custom_functions=custom_functions,
            show_detailed_progress_bar=show_detailed_progress_bar
        )
        self.fncv3_features = FNCV3_FEATURES

    def full_evaluation(
        self,
        dataf: pd.DataFrame,
        pred_cols: List[str],
        target_col: str = "target",
        benchmark_cols: list = None,
    ) -> pd.DataFrame:
        val_stats = pd.DataFrame()
        dataf = dataf.fillna(0.5)
        feature_cols = [col for col in dataf.columns if col.startswith("feature")]

        # Check if sufficient columns are present in dataf to compute FNC
        feature_set = set(dataf.columns)
        if set(self.fncv3_features).issubset(feature_set):
            print(
                "Using 'v4.2/features.json/fncv3_features' feature set to calculate FNC metrics."
            )
            valid_features = self.fncv3_features
        else:
            print(
                "WARNING: No suitable feature set defined for FNC. Skipping calculation of FNC."
            )
            valid_features = []

        with tqdm(pred_cols, desc="Evaluation") as pbar:
            for col in pbar:
                # Metrics that can be calculated for both Numerai Classic and Signals
                col_stats = self.evaluation_one_col(
                    dataf=dataf,
                    feature_cols=feature_cols,
                    pred_col=col,
                    target_col=target_col,
                    benchmark_cols=benchmark_cols,
                )
                # Numerai Classic specific metrics
                if valid_features and "fncv3_mean_std_sharpe" in self.metrics_list:
                    pbar.set_description_str(f"fncv3_mean_std_sharpe for evaluation of '{col}'")
                    # Using only valid features defined in FNCV3_FEATURES
                    fnc_v3, fn_std_v3, fn_sharpe_v3 = self.feature_neutral_mean_std_sharpe(
                        dataf=dataf,
                        pred_col=col,
                        target_col=target_col,
                        feature_names=valid_features,
                    )
                    col_stats.loc[col, "feature_neutral_mean_v3"] = fnc_v3
                    col_stats.loc[col, "feature_neutral_std_v3"] = fn_std_v3
                    col_stats.loc[col, "feature_neutral_sharpe_v3"] = fn_sharpe_v3

                val_stats = pd.concat([val_stats, col_stats], axis=0)
        return val_stats


class NumeraiSignalsEvaluator(BaseEvaluator):
    """Evaluator for all metrics that are relevant in Numerai Signals."""
    # Columns retrievable from Numerai Signals diagnostics.
    # More info: https://forum.numer.ai/t/signals-diagnostics-guide/5950
    VALID_DIAGNOSTICS_COLS = ["validationCorrV4", "validationFncV4", "validationIcV2", "validationRic"]

    def __init__(
        self,
        era_col: str = "friday_date",
        metrics_list: List[str] = FAST_METRICS,
        custom_functions: Dict[str, Dict[str, Any]] = None,
        show_detailed_progress_bar: bool = True,
    ):
        for metric in metrics_list:
            assert (
                metric in ALL_SIGNALS_METRICS
            ), f"Metric '{metric}' not found. Valid metrics: {ALL_SIGNALS_METRICS}."
        super().__init__(
            era_col=era_col, metrics_list=metrics_list, custom_functions=custom_functions,
            show_detailed_progress_bar=show_detailed_progress_bar
        )

    def get_diagnostics(
        self, val_dataf: pd.DataFrame, model_name: str, key: Key, timeout_min: int = 2,
        col: Union[str, None] = "validationFncV4"
    ) -> pd.DataFrame:
        """
        Retrieved neutralized validation correlation by era. \n
        Calculated on Numerai servers. \n
        :param val_dataf: A DataFrame containing prediction, friday_date, ticker and data_type columns. \n
        data_type column should contain 'validation' instances. \n
        :param model_name: Any model name for which you have authentication credentials. \n
        :param key: Key object to authenticate upload of diagnostics. \n
        :param timeout_min: How many minutes to wait on diagnostics Computing on Numerai servers before timing out. \n
        :param col: Which column to return. Should be one of ['validationCorrV4', 'validationFncV4', 'validationIcV2', 'validationRic']. If None, all columns will be returned. \n
        2 minutes by default. \n
        :return: Pandas Series with era as index and neutralized validation correlations (validationCorr).
        """
        assert col in self.VALID_DIAGNOSTICS_COLS or col is None, f"corr_col should be one of {self.VALID_DIAGNOSTICS_COLS} or None. Got: '{col}'"
        api = SignalsAPI(public_id=key.pub_id, secret_key=key.secret_key)
        model_id = api.get_models()[model_name]
        diagnostics_id = api.upload_diagnostics(df=val_dataf, model_id=model_id)
        data = self.__await_diagnostics(
            api=api,
            model_id=model_id,
            diagnostics_id=diagnostics_id,
            timeout_min=timeout_min,
        )
        diagnostics_df = pd.DataFrame(data["perEraDiagnostics"]).set_index("era")
        diagnostics_df.index = pd.to_datetime(diagnostics_df.index)
        return_cols = [col] if col is not None else self.VALID_DIAGNOSTICS_COLS
        return diagnostics_df[return_cols]

    @staticmethod
    def __await_diagnostics(
        api: SignalsAPI,
        model_id: str,
        diagnostics_id: str,
        timeout_min: int,
        interval_sec: int = 15,
    ):
        """
        Wait for diagnostics to be uploaded.
        Try every 'interval_sec' seconds until 'timeout_min' minutes have passed.
        """
        timeout = time.time() + 60 * timeout_min
        data = {"status": "not_done"}
        while time.time() < timeout:
            data = api.diagnostics(model_id=model_id, diagnostics_id=diagnostics_id)[0]
            if not data["status"] == "done":
                print(
                    f"Diagnostics not processed yet. Sleeping for another {interval_sec} seconds."
                )
                time.sleep(interval_sec)
            else:
                break
        if not data["status"] == "done":
            raise Exception(
                f"Diagnostics couldn't be retrieved within {timeout_min} minutes after uploading. Check if Numerai API is offline."
            )
        return data

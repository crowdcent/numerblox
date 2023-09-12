import time
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Union
from numerapi import SignalsAPI

from .numerframe import NumerFrame, create_numerframe
from .postprocessing import FeatureNeutralizer
from .key import Key
from .features import FNCV3_FEATURES

class BaseEvaluator:
    """
    Evaluation functionality that is relevant for both
    Numerai Classic and Numerai Signals.

    :param era_col: Column name pointing to eras. \n
    Most commonly "era" for Numerai Classic and "friday_date" for Numerai Signals. \n
    :param fast_mode: Will skip compute intensive metrics if set to True,
    namely max_exposure, feature neutral mean, TB200 and TB500.

    Note that we calculate the sample standard deviation with ddof=0. 
    It may differ slightly from the standard Pandas calculation, but 
    is consistent with how NumPy computes standard deviation. 
    More info: 
    https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
    """
    def __init__(self, era_col: str = "era", fast_mode=False):
        self.era_col = era_col
        self.fast_mode = fast_mode

    def full_evaluation(
        self,
        dataf: NumerFrame,
        example_col: str,
        pred_cols: list = None,
        target_col: str = "target",
    ) -> pd.DataFrame:
        """
        Perform evaluation for each prediction column in the NumerFrame
        against give target and example prediction column.
        """
        val_stats = pd.DataFrame()
        cat_cols = dataf.get_feature_data.select_dtypes(include=['category']).columns.to_list()
        if cat_cols:
            rich_print(f":warning: WARNING: Categorical features detected that cannot be used for neutralization. Removing columns: '{cat_cols}' for evaluation. :warning:")
            dataf.loc[:, dataf.feature_cols] = dataf.get_feature_data.select_dtypes(exclude=['category'])
        dataf = dataf.fillna(0.5)
        pred_cols = dataf.prediction_cols if not pred_cols else pred_cols
        for col in tqdm(pred_cols, desc="Evaluation: "):
            col_stats = self.evaluation_one_col(
                dataf=dataf,
                pred_col=col,
                target_col=target_col,
                example_col=example_col,
            )
            val_stats = pd.concat([val_stats, col_stats], axis=0)
        return val_stats

    def evaluation_one_col(
        self,
        dataf: NumerFrame,
        pred_col: str,
        target_col: str,
        example_col: str,
    ):
        """
        Perform evaluation for one prediction column
        against given target and example prediction column.
        """
        col_stats = pd.DataFrame()
        # Compute stats
        val_corrs = self.per_era_corrs(
            dataf=dataf, pred_col=pred_col, target_col=target_col
        )
        val_numerai_corrs = self.per_era_numerai_corrs(
            dataf=dataf, pred_col=pred_col, target_col=target_col
        )
        mean, std, sharpe = self.mean_std_sharpe(era_corrs=val_numerai_corrs)
        legacy_mean, legacy_std, legacy_sharpe = self.mean_std_sharpe(era_corrs=val_corrs)
        max_drawdown = self.max_drawdown(era_corrs=val_numerai_corrs)
        apy = self.apy(era_corrs=val_numerai_corrs)
        example_corr = self.example_correlation(
            dataf=dataf, pred_col=pred_col, example_col=example_col
        )
        # Max. drawdown should be have an additional minus because we use negative numbers for max. drawdown.
        calmar = apy / -max_drawdown

        col_stats.loc[pred_col, "target"] = target_col
        col_stats.loc[pred_col, "mean"] = mean
        col_stats.loc[pred_col, "std"] = std
        col_stats.loc[pred_col, "sharpe"] = sharpe
        col_stats.loc[pred_col, "max_drawdown"] = max_drawdown
        col_stats.loc[pred_col, "apy"] = apy
        col_stats.loc[pred_col, "calmar_ratio"] = calmar
        col_stats.loc[pred_col, "corr_with_example_preds"] = example_corr
        col_stats.loc[pred_col, "legacy_mean"] = legacy_mean
        col_stats.loc[pred_col, "legacy_std"] = legacy_std
        col_stats.loc[pred_col, "legacy_sharpe"] = legacy_sharpe

        # Compute intensive stats
        if not self.fast_mode:
            max_feature_exposure = self.max_feature_exposure(
                dataf=dataf, pred_col=pred_col
            )
            fn_mean, fn_std, fn_sharpe = self.feature_neutral_mean_std_sharpe(
                dataf=dataf, pred_col=pred_col, target_col=target_col
            )
            tb200_mean, tb200_std, tb200_sharpe = self.tbx_mean_std_sharpe(
                dataf=dataf, pred_col=pred_col, target_col=target_col, tb=200
            )
            tb500_mean, tb500_std, tb500_sharpe = self.tbx_mean_std_sharpe(
                dataf=dataf, pred_col=pred_col, target_col=target_col, tb=500
            )
            ex_diss = self.exposure_dissimilarity(
                dataf=dataf, pred_col=pred_col, example_col=example_col
            )

            col_stats.loc[pred_col, "max_feature_exposure"] = max_feature_exposure
            col_stats.loc[pred_col, "feature_neutral_mean"] = fn_mean
            col_stats.loc[pred_col, "feature_neutral_std"] = fn_std
            col_stats.loc[pred_col, "feature_neutral_sharpe"] = fn_sharpe
            col_stats.loc[pred_col, "tb200_mean"] = tb200_mean
            col_stats.loc[pred_col, "tb200_std"] = tb200_std
            col_stats.loc[pred_col, "tb200_sharpe"] = tb200_sharpe
            col_stats.loc[pred_col, "tb500_mean"] = tb500_mean
            col_stats.loc[pred_col, "tb500_std"] = tb500_std
            col_stats.loc[pred_col, "tb500_sharpe"] = tb500_sharpe
            col_stats.loc[pred_col, "exposure_dissimilarity"] = ex_diss
        return col_stats

    def per_era_corrs(
        self, dataf: pd.DataFrame, pred_col: str, target_col: str
    ) -> pd.Series:
        """Correlation between prediction and target for each era."""
        return dataf.groupby(dataf[self.era_col]).apply(
            lambda d: self._normalize_uniform(d[pred_col].fillna(0.5)).corr(
                d[target_col]
            )
        )
    
    def per_era_numerai_corrs(
            self, dataf: pd.DataFrame, pred_col: str, target_col: str
        ) -> pd.Series:
        """Numerai Corr between prediction and target for each era."""
        return dataf.groupby(dataf[self.era_col]).apply(
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
        sharpe = mean / std
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
        ranked_preds = self._normalize_uniform(dataf[pred_col].fillna(0.5), 
                                               method="average")
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
            payout_product ** (
                # 52 weeks of compounding minus n for stake compounding lag
                (52 - stake_compounding_lag) / len(payout_scores))
            - 1
        ) * 100

    def example_correlation(
        self, dataf: Union[pd.DataFrame, NumerFrame], pred_col: str, example_col: str
    ):
        """Correlations with example predictions."""
        return self.per_era_corrs(
            dataf=dataf,
            pred_col=pred_col,
            target_col=example_col,
        ).mean()

    def max_feature_exposure(
        self, dataf: Union[pd.DataFrame, NumerFrame], pred_col: str
    ) -> np.float64:
        """Maximum exposure over all features."""
        max_per_era = dataf.groupby(self.era_col).apply(
            lambda d: d[dataf.feature_cols].corrwith(d[pred_col]).abs().max()
        )
        max_feature_exposure = max_per_era.mean(skipna=True)
        return max_feature_exposure

    def feature_neutral_mean_std_sharpe(
        self, dataf: Union[pd.DataFrame, NumerFrame], pred_col: str, target_col: str, feature_names: list = None
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Feature neutralized mean performance.
        More info: https://docs.numer.ai/tournament/feature-neutral-correlation
        """
        fn = FeatureNeutralizer(pred_name=pred_col,
                                feature_names=feature_names,
                                proportion=1.0)
        neutralized_dataf = fn(dataf=dataf)
        neutral_corrs = self.per_era_numerai_corrs(
            dataf=neutralized_dataf,
            pred_col=f"{pred_col}_neutralized_1.0",
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

    def exposure_dissimilarity(self, dataf: NumerFrame, pred_col: str, example_col: str) -> np.float32:
        """
        Model pattern of feature exposure to the example column.
        See TC details forum post: https://forum.numer.ai/t/true-contribution-details/5128/4
        """
        U = dataf.get_feature_data.corrwith(dataf[pred_col]).values
        E = dataf.get_feature_data.corrwith(dataf[example_col]).values
        exp_dis = 1 - np.dot(U, E) / np.dot(E, E)
        return exp_dis


    @staticmethod
    def _neutralize_series(series: pd.Series, by: pd.Series, proportion=1.0) -> pd.Series:
        scores = series.values.reshape(-1, 1)
        exposures = by.values.reshape(-1, 1)

        # This line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
        exposures = np.hstack(
            (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1))
        )

        correction = proportion * (
            exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0])
        )
        corrected_scores = scores - correction
        neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
        return neutralized

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
        """Normalize predictions uniformly using ranks."""
        x = (df.rank(method=method) - 0.5) / len(df)
        return pd.Series(x, index=df.index)

    def plot_correlations(
        self,
        dataf: NumerFrame,
        pred_cols: list = None,
        corr_cols: list = None,
        target_col: str = "target",
        roll_mean: int = 20,
    ):
        """
        Plot per era correlations over time.
        :param dataf: NumerFrame that contains at least all pred_cols, target_col and corr_cols.
        :param pred_cols: List of prediction columns that per era correlations of and plot.
        By default, all predictions_cols in the NumerFrame are used. 
        :param corr_cols: Per era correlations already prepared to include in the plot.
        This is optional for if you already have per era correlations prepared in your input dataf.
        :param target_col: Target column name to compute per era correlations against.
        :param roll_mean: How many eras should be averaged to compute a rolling score.
        """
        validation_by_eras = pd.DataFrame()
        pred_cols = dataf.prediction_cols if not pred_cols else pred_cols
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


class NumeraiClassicEvaluator(BaseEvaluator):
    """Evaluator for all metrics that are relevant in Numerai Classic."""
    def __init__(self, era_col: str = "era", fast_mode=False):
        super().__init__(era_col=era_col, fast_mode=fast_mode)
        self.fncv3_features = FNCV3_FEATURES

    def full_evaluation(
        self,
        dataf: NumerFrame,
        example_col: str,
        pred_cols: list = None,
        target_col: str = "target",
    ) -> pd.DataFrame:
        val_stats = pd.DataFrame()
        dataf = dataf.fillna(0.5)
        pred_cols = dataf.prediction_cols if not pred_cols else pred_cols

        # Check if sufficient columns are present in dataf to compute FNC
        feature_set = set(dataf.columns)
        if set(self.fncv3_features).issubset(feature_set):
            print("Using 'v4.2/features.json/fncv3_features' feature set to calculate FNC metrics.")
            valid_features = self.fncv3_features
        else:
            print("WARNING: No suitable feature set defined for FNC. Skipping calculation of FNC.")
            valid_features = []

        for col in tqdm(pred_cols, desc="Evaluation: "):
            # Metrics that can be calculated for both Numerai Classic and Signals
            col_stats = self.evaluation_one_col(
                dataf=dataf,
                pred_col=col,
                target_col=target_col,
                example_col=example_col,
            )
            # Numerai Classic specific metrics
            if not self.fast_mode and valid_features:
                fnc_v3, fn_std_v3, fn_sharpe_v3 = self.feature_neutral_mean_std_sharpe(
                    dataf=dataf, pred_col=col, target_col=target_col, feature_names=valid_features
                )
                col_stats.loc[col, "feature_neutral_mean_v3"] = fnc_v3
                col_stats.loc[col, "feature_neutral_std_v3"] = fn_std_v3
                col_stats.loc[col, "feature_neutral_sharpe_v3"] = fn_sharpe_v3
            val_stats = pd.concat([val_stats, col_stats], axis=0)
        return val_stats

# %% ../nbs/07_evaluation.ipynb 14
class NumeraiSignalsEvaluator(BaseEvaluator):
    """Evaluator for all metrics that are relevant in Numerai Signals."""
    def __init__(self, era_col: str = "friday_date", fast_mode=False):
        super().__init__(era_col=era_col, fast_mode=fast_mode)

    def get_neutralized_corr(self, val_dataf: pd.DataFrame, model_name: str, key: Key, timeout_min: int = 2) -> pd.Series:
        """
        Retrieved neutralized validation correlation by era. \n
        Calculated on Numerai servers. \n
        :param val_dataf: A DataFrame containing prediction, friday_date, ticker and data_type columns. \n
        data_type column should contain 'validation' instances. \n
        :param model_name: Any model name for which you have authentication credentials. \n
        :param key: Key object to authenticate upload of diagnostics. \n
        :param timeout_min: How many minutes to wait on diagnostics processing on Numerai servers before timing out. \n
        2 minutes by default. \n
        :return: Pandas Series with era as index and neutralized validation correlations (validationCorr).
        """
        api = SignalsAPI(public_id=key.pub_id, secret_key=key.secret_key)
        model_id = api.get_models()[model_name]
        diagnostics_id = api.upload_diagnostics(df=val_dataf, model_id=model_id)
        data = self.__await_diagnostics(api=api, model_id=model_id, diagnostics_id=diagnostics_id, timeout_min=timeout_min)
        dataf = pd.DataFrame(data['perEraDiagnostics']).set_index("era")['validationCorr']
        dataf.index = pd.to_datetime(dataf.index)
        return dataf

    @staticmethod
    def __await_diagnostics(api: SignalsAPI, model_id: str, diagnostics_id: str, timeout_min: int, interval_sec: int = 15):
        """
        Wait for diagnostics to be uploaded.
        Try every 'interval_sec' seconds until 'timeout_min' minutes have passed.
        """
        timeout = time.time() + 60 * timeout_min
        data = {"status": "not_done"}
        while time.time() < timeout:
            data = api.diagnostics(model_id=model_id, diagnostics_id=diagnostics_id)[0]
            if not data['status'] == 'done':
                print(f"Diagnostics not processed yet. Sleeping for another {interval_sec} seconds.")
                time.sleep(interval_sec)
            else:
                break
        if not data['status'] == 'done':
            raise Exception(f"Diagnostics couldn't be retrieved within {timeout_min} minutes after uploading. Check if Numerai API is offline.")
        return data

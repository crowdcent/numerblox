import os
import time
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import pandas_ta as ta
from tqdm.auto import tqdm
from functools import wraps
from scipy.stats import rankdata
from abc import ABC, abstractmethod
from rich import print as rich_print
from typing import Union, Tuple, List
from multiprocessing.pool import Pool
from sklearn.linear_model import Ridge
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array


from .numerframe import NumerFrame
from .features import V4_2_FEATURE_GROUP_MAPPING

class BaseProcessor(BaseEstimator, TransformerMixin):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> NumerFrame:
        ...

    def __call__(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> NumerFrame:
        return self.transform(X=X, y=y, **kwargs)


class CopyPreProcessor(BaseProcessor):
    """Copy DataFrame to avoid manipulation of original DataFrame."""

    def __init__(self):
        super().__init__()

    def transform(self, X: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        return NumerFrame(X.copy())


class FeatureSelectionPreProcessor(BaseProcessor):
    """
    Keep only features given + all target, predictions and aux columns.
    """

    def __init__(self, feature_cols: Union[str, list]):
        super().__init__()
        self.feature_cols = feature_cols

    def transform(self, X: NumerFrame, y=None) -> NumerFrame:
        keep_cols = (
            self.feature_cols
            + X.target_cols
            + X.prediction_cols
            + X.aux_cols
        )
        X = X.loc[:, keep_cols]
        return NumerFrame(X)


class TargetSelectionPreProcessor(BaseProcessor):
    """
    Keep only features given + all target, predictions and aux columns.
    """

    def __init__(self, target_cols: Union[str, list]):
        super().__init__()
        self.target_cols = target_cols

    def transform(self, dataf: NumerFrame) -> NumerFrame:
        keep_cols = (
            self.target_cols
            + dataf.feature_cols
            + dataf.prediction_cols
            + dataf.aux_cols
        )
        dataf = dataf.loc[:, keep_cols]
        return NumerFrame(dataf)


class ReduceMemoryProcessor(BaseProcessor):
    """
    Reduce memory usage as much as possible.

    Credits to kainsama and others for writing about memory usage reduction for Numerai data:
    https://forum.numer.ai/t/reducing-memory/313

    :param deep_mem_inspect: Introspect the data deeply by interrogating object dtypes.
    Yields a more accurate representation of memory usage if you have complex object columns.
    """

    def __init__(self, deep_mem_inspect=False):
        super().__init__()
        self.deep_mem_inspect = deep_mem_inspect

    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        dataf = self._reduce_mem_usage(dataf)
        return NumerFrame(dataf)

    def _reduce_mem_usage(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all columns and modify the numeric column types
        to reduce memory usage.
        """
        start_memory_usage = (
            dataf.memory_usage(deep=self.deep_mem_inspect).sum() / 1024**2
        )
        rich_print(
            f"Memory usage of DataFrame is [bold]{round(start_memory_usage, 2)} MB[/bold]"
        )

        for col in dataf.columns:
            col_type = dataf[col].dtype.name

            if col_type not in [
                "object",
                "category",
                "datetime64[ns, UTC]",
                "datetime64[ns]",
            ]:
                c_min = dataf[col].min()
                c_max = dataf[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataf[col] = dataf[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        dataf[col] = dataf[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        dataf[col] = dataf[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        dataf[col] = dataf[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        dataf[col] = dataf[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        dataf[col] = dataf[col].astype(np.float32)
                    else:
                        dataf[col] = dataf[col].astype(np.float64)

        end_memory_usage = (
            dataf.memory_usage(deep=self.deep_mem_inspect).sum() / 1024**2
        )
        rich_print(
            f"Memory usage after optimization is: [bold]{round(end_memory_usage, 2)} MB[/bold]"
        )
        rich_print(
            f"[green] Usage decreased by [bold]{round(100 * (start_memory_usage - end_memory_usage) / start_memory_usage, 2)}%[/bold][/green]"
        )
        return dataf


class BayesianGMMTargetProcessor(BaseProcessor):
    """
    Generate synthetic (fake) target using a Bayesian Gaussian Mixture model. \n
    Based on Michael Oliver's GitHub Gist implementation: \n
    https://gist.github.com/the-moliver/dcdd2862dc2c78dda600f1b449071c93

    :param target_col: Column from which to create fake target. \n
    :param feature_names: Selection of features used for Bayesian GMM. All features by default.
    :param n_components: Number of components for fitting Bayesian Gaussian Mixture Model.
    """

    def __init__(
        self,
        target_col: str = "target",
        feature_names: list = None,
        n_components: int = 6,
    ):
        super().__init__()
        self.target_col = target_col
        self.feature_names = feature_names
        self.n_components = n_components
        self.ridge = Ridge(fit_intercept=False)
        self.bins = [0, 0.05, 0.25, 0.75, 0.95, 1]

    def transform(self, dataf: NumerFrame, *args, **kwargs) -> NumerFrame:
        all_eras = dataf[dataf.meta.era_col].unique()
        coefs = self._get_coefs(dataf=dataf, all_eras=all_eras)
        bgmm = self._fit_bgmm(coefs=coefs)
        fake_target = self._generate_target(dataf=dataf, bgmm=bgmm, all_eras=all_eras)
        dataf[f"{self.target_col}_fake"] = fake_target
        return NumerFrame(dataf)

    def _get_coefs(self, dataf: NumerFrame, all_eras: list) -> np.ndarray:
        """
        Generate coefficients for BGMM.
        Data should already be scaled between 0 and 1
        (Already done with Numerai Classic data)
        """
        coefs = []
        for era in all_eras:
            features, target = self.__get_features_target(dataf=dataf, era=era)
            self.ridge.fit(features, target)
            coefs.append(self.ridge.coef_)
        stacked_coefs = np.vstack(coefs)
        return stacked_coefs

    def _fit_bgmm(self, coefs: np.ndarray) -> BayesianGaussianMixture:
        """
        Fit Bayesian Gaussian Mixture model on coefficients and normalize.
        """
        bgmm = BayesianGaussianMixture(n_components=self.n_components)
        bgmm.fit(coefs)
        # make probability of sampling each component equal to better balance rare regimes
        bgmm.weights_[:] = 1 / self.n_components
        return bgmm

    def _generate_target(
        self, dataf: NumerFrame, bgmm: BayesianGaussianMixture, all_eras: list
    ) -> np.ndarray:
        """Generate fake target using Bayesian Gaussian Mixture model."""
        fake_target = []
        for era in tqdm(all_eras, desc="Generating fake target"):
            features, _ = self.__get_features_target(dataf=dataf, era=era)
            # Sample a set of weights from GMM
            beta, _ = bgmm.sample(1)
            # Create fake continuous target
            fake_targ = features @ beta[0]
            # Bin fake target like real target
            fake_targ = (rankdata(fake_targ) - 0.5) / len(fake_targ)
            fake_targ = (np.digitize(fake_targ, self.bins) - 1) / 4
            fake_target.append(fake_targ)
        return np.concatenate(fake_target)

    def __get_features_target(self, dataf: NumerFrame, era) -> tuple:
        """Get features and target for one era and center data."""
        sub_df = dataf[dataf[dataf.meta.era_col] == era]
        features = self.feature_names if self.feature_names else sub_df.feature_cols
        target = sub_df[self.target_col].values - 0.5
        features = sub_df[features].values - 0.5
        return features, target


class GroupStatsPreProcessor(BaseProcessor):
    """
    WARNING: Only supported for v4.2 (Rain) data. The Rain dataset (re)introduced feature groups. \n
    
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
        self.group_names = groups if groups else self.all_groups
        self.feature_group_mapping = V4_2_FEATURE_GROUP_MAPPING

    def transform(self, dataf: pd.DataFrame, *args, **kwargs) -> NumerFrame:
        """Check validity and add group features."""
        dataf = dataf.pipe(self._add_group_features)
        return NumerFrame(dataf)

    def _add_group_features(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """Mean, standard deviation and skew for each group."""
        dataf = dataf.copy()
        for group in self.group_names:
            cols = self.feature_group_mapping[group]
            dataf.loc[:, f"feature_{group}_mean"] = dataf[cols].mean(axis=1)
            dataf.loc[:, f"feature_{group}_std"] = dataf[cols].std(axis=1)
            dataf.loc[:, f"feature_{group}_skew"] = dataf[cols].skew(axis=1)
            
        return dataf


class KatsuFeatureGenerator(BaseProcessor):
    """
    Effective feature engineering setup based on Katsu's starter notebook.
    Based on source by Katsu1110: https://www.kaggle.com/code1110/numeraisignals-starter-for-beginners

    :param windows: Time interval to apply for window features: \n
    1. Percentage Rate of change \n
    2. Volatility \n
    3. Moving Average gap \n
    :param ticker_col: Columns with tickers to iterate over. \n
    :param close_col: Column name where you have closing price stored.
    """

    warnings.filterwarnings("ignore")

    def __init__(
        self,
        windows: list,
        ticker_col: str = "ticker",
        close_col: str = "close",
        num_cores: int = None,
    ):
        super().__init__()
        self.windows = windows
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.num_cores = num_cores if num_cores else os.cpu_count()

    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """Multiprocessing feature engineering."""
        tickers = dataf.loc[:, self.ticker_col].unique().tolist()
        rich_print(
            f"Feature engineering for {len(tickers)} tickers using {self.num_cores} CPU cores."
        )
        dataf_list = [
            x
            for _, x in tqdm(
                dataf.groupby(self.ticker_col), desc="Generating ticker DataFrames"
            )
        ]
        dataf = self._generate_features(dataf_list=dataf_list)
        return NumerFrame(dataf)

    def feature_engineering(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for single ticker."""
        close_series = dataf.loc[:, self.close_col]
        for x in self.windows:
            dataf.loc[
                :, f"feature_{self.close_col}_ROCP_{x}"
            ] = close_series.pct_change(x)

            dataf.loc[:, f"feature_{self.close_col}_VOL_{x}"] = (
                np.log1p(close_series).pct_change().rolling(x).std()
            )

            dataf.loc[:, f"feature_{self.close_col}_MA_gap_{x}"] = (
                close_series / close_series.rolling(x).mean()
            )

        dataf.loc[:, "feature_RSI"] = self._rsi(close_series)
        macd, macd_signal = self._macd(close_series)
        dataf.loc[:, "feature_MACD"] = macd
        dataf.loc[:, "feature_MACD_signal"] = macd_signal
        return dataf.bfill()

    def _generate_features(self, dataf_list: list) -> pd.DataFrame:
        """Add features for list of ticker DataFrames and concatenate."""
        with Pool(self.num_cores) as p:
            feature_datafs = list(
                tqdm(
                    p.imap(self.feature_engineering, dataf_list),
                    desc="Generating features",
                    total=len(dataf_list),
                )
            )
        return pd.concat(feature_datafs)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        See source https://github.com/peerchemist/finta
        and fix https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
        """
        delta = close.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        gain = up.ewm(com=(period - 1), min_periods=period).mean()
        loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        rs = gain / loss
        return pd.Series(100 - (100 / (1 + rs)))

    def _macd(
        self, close: pd.Series, span1=12, span2=26, span3=9
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD and MACD signal."""
        exp1 = self.__ema1(close, span1)
        exp2 = self.__ema1(close, span2)
        macd = 100 * (exp1 - exp2) / exp2
        signal = self.__ema1(macd, span3)
        return macd, signal

    @staticmethod
    def __ema1(series: pd.Series, span: int) -> pd.Series:
        """Exponential moving average"""
        a = 2 / (span + 1)
        return series.ewm(alpha=a).mean()


class EraQuantileProcessor(BaseProcessor):
    def __init__(
        self,
        num_quantiles: int = 50,
        era_col: str = "date",
        features: list = None,
        random_state: int = 0
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.era_col = era_col
        self.features = features 
        self.random_state = random_state

    def _process_feature(self, group_data: pd.Series, feature: str) -> pd.Series:
        quantizer = QuantileTransformer(
            n_quantiles=self.num_quantiles, random_state=self.random_state
        )
        transformed_data = quantizer.fit_transform(group_data.to_frame()).ravel()
        return pd.Series(transformed_data, index=group_data.index)


    def transform(
        self,
        dataf: Union[pd.DataFrame, NumerFrame],
    ) -> pd.DataFrame:
        features = self.features if self.features else [col for col in dataf.columns if col != self.era_col]
        print(f"Quantiling for {len(features)} features.")

        date_groups = dataf.groupby(self.era_col, group_keys=False)
        for feature in tqdm(features):
            group_data = date_groups[feature].apply(lambda x: self._process_feature(x, feature))
            dataf[f"{feature}_quantile{self.num_quantiles}"] = group_data
        return dataf


class TickerMapper(BaseProcessor):
    """
    Map ticker from one format to another. \n
    :param ticker_col: Column used for mapping. Must already be present in the input data. \n
    :param target_ticker_format: Format to map tickers to. Must be present in the ticker map. \n
    For default mapper supported ticker formats are: ['ticker', 'bloomberg_ticker', 'yahoo'] \n
    :param mapper_path: Path to CSV file containing at least ticker_col and target_ticker_format columns. \n
    Can be either a web link of local path. Numerai Signals mapping by default.
    """

    def __init__(
        self, ticker_col: str = "ticker", target_ticker_format: str = "bloomberg_ticker",
        mapper_path: str = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv"
    ):
        super().__init__()
        self.ticker_col = ticker_col
        self.target_ticker_format = target_ticker_format

        self.signals_map_path = mapper_path
        self.ticker_map = pd.read_csv(self.signals_map_path)

        assert (
            self.ticker_col in self.ticker_map.columns
        ), f"Ticker column '{self.ticker_col}' is not available in ticker mapping."
        assert (
            self.target_ticker_format in self.ticker_map.columns
        ), f"Target ticker column '{self.target_ticker_format}' is not available in ticker mapping."

        self.mapping = dict(
            self.ticker_map[[self.ticker_col, self.target_ticker_format]].values
        )

    def transform(
        self, dataf: Union[pd.DataFrame, NumerFrame], **kwargs
    ) -> NumerFrame:
        dataf[self.target_ticker_format] = dataf[self.ticker_col].map(self.mapping)
        return NumerFrame(dataf)

# %% ../nbs/03_preprocessing.ipynb 66
class SignalsTargetProcessor(BaseProcessor):
    """
    Engineer targets for Numerai Signals. \n
    More information on implements Numerai Signals targets: \n
    https://forum.numer.ai/t/decoding-the-signals-target/2501

    :param price_col: Column from which target will be derived. \n
    :param windows: Timeframes to use for engineering targets. 10 and 20-day by default. \n
    :param bins: Binning used to create group targets. Nomi binning by default. \n
    :param labels: Scaling for binned target. Must be same length as resulting bins (bins-1). Numerai labels by default.
    """

    def __init__(
        self,
        price_col: str = "close",
        windows: list = None,
        bins: list = None,
        labels: list = None,
    ):
        super().__init__()
        self.price_col = price_col
        self.windows = windows if windows else [10, 20]
        self.bins = bins if bins else [0, 0.05, 0.25, 0.75, 0.95, 1]
        self.labels = labels if labels else [0, 0.25, 0.50, 0.75, 1]

    def transform(self, dataf: NumerFrame) -> NumerFrame:
        for window in tqdm(self.windows, desc="Signals target engineering windows"):
            dataf.loc[:, f"target_{window}d_raw"] = (
                dataf[self.price_col].pct_change(periods=window).shift(-window)
            )
            era_groups = dataf.groupby(dataf.meta.era_col)

            dataf.loc[:, f"target_{window}d_rank"] = era_groups[
                f"target_{window}d_raw"
            ].rank(pct=True, method="first")
            dataf.loc[:, f"target_{window}d_group"] = era_groups[
                f"target_{window}d_rank"
            ].transform(
                lambda group: pd.cut(
                    group, bins=self.bins, labels=self.labels, include_lowest=True
                )
            )
        return NumerFrame(dataf)


class LagPreProcessor(BaseProcessor):
    """
    Add lag features based on given windows.

    :param windows: All lag windows to process for all features. \n
    [5, 10, 15, 20] by default (4 weeks lookback) \n
    :param ticker_col: Column name for grouping by tickers. \n
    :param feature_names: All features for which you want to create lags. All features by default.
    """

    def __init__(
        self,
        windows: list = None,
        ticker_col: str = "bloomberg_ticker",
        feature_names: list = None,
    ):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]
        self.ticker_col = ticker_col
        self.feature_names = feature_names

    def transform(self, dataf: NumerFrame) -> NumerFrame:
        feature_names = self.feature_names if self.feature_names else dataf.feature_cols
        ticker_groups = dataf.groupby(self.ticker_col)
        for feature in tqdm(feature_names, desc="Lag feature generation"):
            feature_group = ticker_groups[feature]
            for day in self.windows:
                shifted = feature_group.shift(day, axis=0)
                dataf.loc[:, f"{feature}_lag{day}"] = shifted
        return NumerFrame(dataf)


class DifferencePreProcessor(BaseProcessor):
    """
    Add difference features based on given windows. Run LagPreProcessor first.

    :param windows: All lag windows to process for all features. \n
    :param feature_names: All features for which you want to create differences. All features that also have lags by default. \n
    :param pct_change: Method to calculate differences. If True, will calculate differences with a percentage change. Otherwise calculates a simple difference. Defaults to False \n
    :param abs_diff: Whether to also calculate the absolute value of all differences. Defaults to True \n
    """

    def __init__(
        self,
        windows: list = None,
        feature_names: list = None,
        pct_diff: bool = False,
        abs_diff: bool = False,
    ):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]
        self.feature_names = feature_names
        self.pct_diff = pct_diff
        self.abs_diff = abs_diff

    def transform(self, dataf: NumerFrame) -> NumerFrame:
        feature_names = self.feature_names if self.feature_names else dataf.feature_cols
        for feature in tqdm(feature_names, desc="Difference feature generation"):
            lag_columns = dataf.get_pattern_data(f"{feature}_lag").columns
            if not lag_columns.empty:
                for day in self.windows:
                    differenced_values = (
                        (dataf[feature] / dataf[f"{feature}_lag{day}"]) - 1
                        if self.pct_diff
                        else dataf[feature] - dataf[f"{feature}_lag{day}"]
                    )
                    dataf[f"{feature}_diff{day}"] = differenced_values
                    if self.abs_diff:
                        dataf[f"{feature}_absdiff{day}"] = np.abs(
                            dataf[f"{feature}_diff{day}"]
                        )
            else:
                rich_print(
                    f":warning: WARNING: Skipping {feature}. Lag features for feature: {feature} were not detected. Have you already run LagPreProcessor? :warning:"
                )
        return NumerFrame(dataf)


class PandasTaFeatureGenerator(BaseProcessor):
    """
    Generate features with pandas-ta.
    https://github.com/twopirllc/pandas-ta

    :param strategy: Valid Pandas Ta strategy. \n
    For more information on creating a strategy, see: \n
    https://github.com/twopirllc/pandas-ta#pandas-ta-strategy \n
    By default, a strategy with RSI(14) and RSI(60) is used. \n
    :param ticker_col: Column name for grouping by tickers. \n
    :param num_cores: Number of cores to use for multiprocessing. \n
    By default, all available cores are used. \n
    """
    def __init__(self, 
                 strategy: ta.Strategy = None,
                 ticker_col: str = "ticker",
                 num_cores: int = None,
    ):
        super().__init__()
        self.ticker_col = ticker_col
        self.num_cores = num_cores if num_cores else os.cpu_count()
        standard_strategy = ta.Strategy(name="standard", 
                                        ta=[{"kind": "rsi", "length": 14, "col_names": ("feature_RSI_14")},
                                            {"kind": "rsi", "length": 60, "col_names": ("feature_RSI_60")}])
        self.strategy = strategy if strategy is not None else standard_strategy

    def transform(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """
        Main feature generation method. \n 
        :param dataf: DataFrame with columns: [ticker, date, open, high, low, close, volume] \n
        :return: DataFrame with features added.
        """
        dataf_list = [
            x
            for _, x in tqdm(
                dataf.groupby(self.ticker_col), desc="Generating ticker DataFrames"
            )
        ]
        dataf = self._generate_features(dataf_list=dataf_list)
        return NumerFrame(dataf)
    
    def _generate_features(self, dataf_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Add features for list of ticker DataFrames and concatenate.
        :param dataf_list: List of DataFrames for each ticker.
        :return: Concatenated DataFrame for all full list with features added.
        """
        with Pool(self.num_cores) as p:
            feature_datafs = list(
                tqdm(
                    p.imap(self.add_features, dataf_list),
                    desc="Generating pandas-ta features",
                    total=len(dataf_list),
                )
            )
        return pd.concat(feature_datafs)

    def add_features(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """ 
        The TA strategy is applied to the DataFrame here.
        :param ticker_df: DataFrame for a single ticker.
        :return: DataFrame with features added.
        """
        # We use a different multiprocessing engine so shutting off pandas_ta's multiprocessing
        ticker_df.ta.cores = 0
        ticker_df.ta.strategy(self.strategy)
        return ticker_df


class AwesomePreProcessor(BaseProcessor):
    """ TEMPLATE - Do some awesome preprocessing. """
    def __init__(self):
        super().__init__()

    def transform(self, dataf: NumerFrame, **kwargs) -> NumerFrame:
        # Do processing
        ...
        # Parse all contents of NumerFrame to the next pipeline step
        return NumerFrame(dataf)

import os
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm.auto import tqdm
from scipy.stats import rankdata
from abc import abstractmethod
from rich import print as rich_print
from typing import Union, Tuple, List
from multiprocessing.pool import Pool
from sklearn.linear_model import Ridge
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from .numerframe import NumerFrame
from .features import V4_2_FEATURE_GROUP_MAPPING

# Ignore Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class BasePreProcessor(BaseEstimator, TransformerMixin):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        ...

    def __call__(
        self, X: Union[pd.DataFrame, NumerFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        return self.transform(X=X, y=y, **kwargs)
    
    @abstractmethod
    def get_feature_names_out(self, input_features=None) -> List[str]:
        ...
    

class ReduceMemoryProcessor(BasePreProcessor):
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

    def transform(self, dataf: pd.DataFrame) -> pd.DataFrame:
        return self._reduce_mem_usage(dataf)

    def _reduce_mem_usage(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all columns and modify the numeric column types
        to reduce memory usage.
        """
        self.output_cols = dataf.columns.tolist()
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
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return self.output_cols if not input_features else input_features


class BayesianGMMTargetProcessor(BasePreProcessor):
    """
    Generate synthetic (fake) target using a Bayesian Gaussian Mixture model. \n
    Based on Michael Oliver's GitHub Gist implementation: \n
    https://gist.github.com/the-moliver/dcdd2862dc2c78dda600f1b449071c93

    :param n_components: Number of components for fitting Bayesian Gaussian Mixture Model.
    """
    def __init__(
        self,
        n_components: int = 3,
    ):
        super().__init__()
        self.n_components = n_components
        self.ridge = Ridge(fit_intercept=False)
        self.bins = [0, 0.05, 0.25, 0.75, 0.95, 1]

    def fit(self, X: pd.DataFrame, y: pd.Series, eras: pd.Series):
        """
        Fit Bayesian Gaussian Mixture model on coefficients and normalize.
        :param X: DataFrame containing features.
        :param y: Series containing real target.
        :param eras: Series containing era information.
        """
        bgmm = BayesianGaussianMixture(n_components=self.n_components)
        coefs = self._get_coefs(dataf=X, y=y, eras=eras)
        bgmm.fit(coefs)
        # make probability of sampling each component equal to better balance rare regimes
        bgmm.weights_[:] = 1 / self.n_components
        self.bgmm_ = bgmm
        return self

    def transform(self, X: pd.DataFrame, eras: pd.Series) -> np.array:
        """
        Main method for generating fake target.
        :param X: DataFrame containing features.
        :param eras: Series containing era information.
        """
        check_is_fitted(self, "bgmm_")
        assert len(X) == len(eras), "X and eras must be same length."
        all_eras = eras.unique().tolist()
        # Scale data between 0 and 1
        X = X.astype(float)
        X /= X.max()
        X -= 0.5
        X.loc[:, 'era'] = eras

        fake_target = self._generate_target(dataf=X, all_eras=all_eras)
        return fake_target

    def _get_coefs(self, dataf: pd.DataFrame, y: pd.Series, eras: pd.Series) -> np.ndarray:
        """
        Generate coefficients for BGMM.
        :param dataf: DataFrame containing features.
        :param y: Series containing real target.
        """
        coefs = []
        dataf.loc[:, 'era'] = eras
        dataf.loc[:, 'target'] = y
        all_eras = dataf['era'].unique().tolist()
        for era in all_eras:
            era_df = dataf[dataf['era'] == era]
            era_y = era_df.loc[:, 'target']
            era_df = era_df.drop(columns=["era", "target"])
            self.ridge.fit(era_df, era_y)
            coefs.append(self.ridge.coef_)
        stacked_coefs = np.vstack(coefs)
        return stacked_coefs

    def _generate_target(
        self, dataf: pd.DataFrame, all_eras: list
    ) -> np.ndarray:
        """Generate fake target using Bayesian Gaussian Mixture model."""
        fake_target = []
        for era in tqdm(all_eras, desc="Generating fake target"):
            features = dataf[dataf['era'] == era]
            features = features.drop(columns=["era", "target"])
            # Sample a set of weights from GMM
            beta, _ = self.bgmm_.sample(1)
            # Create fake continuous target
            fake_targ = features @ beta[0]
            # Bin fake target like real target
            fake_targ = (rankdata(fake_targ) - 0.5) / len(fake_targ)
            fake_targ = (np.digitize(fake_targ, self.bins) - 1) / 4
            fake_target.append(fake_targ)
        return np.concatenate(fake_target)
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return ["fake_target"] if not input_features else input_features


class GroupStatsPreProcessor(BasePreProcessor):
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
        self.groups = groups 
        self.group_names = groups if self.groups else self.all_groups
        self.feature_group_mapping = V4_2_FEATURE_GROUP_MAPPING

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check validity and add group features."""
        dataf = self._add_group_features(X)
        return dataf

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


class KatsuFeatureGenerator(BasePreProcessor):
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

    def transform(self, dataf: pd.DataFrame) -> pd.DataFrame:
        """
        Multiprocessing feature engineering.
        
        :param dataf: DataFrame with columns: [ticker, date, open, high, low, close, volume] \n
        """
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
        output_cols = self.get_feature_names_out()
        return dataf[output_cols]

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
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        if not input_features:
            feature_names = []
            for x in self.windows:
                feature_names += [
                    f"feature_{self.close_col}_ROCP_{x}",
                    f"feature_{self.close_col}_VOL_{x}",
                    f"feature_{self.close_col}_MA_gap_{x}",
                ]
            feature_names += [
                "feature_RSI",
                "feature_MACD",
                "feature_MACD_signal",
            ]
        else:
            feature_names = input_features
        return feature_names


class EraQuantileProcessor(BasePreProcessor):
    def __init__(
        self,
        num_quantiles: int = 50,
        random_state: int = 0
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.random_state = random_state

    def _process_feature(self, group_data: pd.Series) -> pd.Series:
        quantizer = QuantileTransformer(
            n_quantiles=self.num_quantiles, random_state=self.random_state
        )
        transformed_data = quantizer.fit_transform(group_data.to_frame()).ravel()
        return pd.Series(transformed_data, index=group_data.index)

    def transform(
        self, dataf: pd.DataFrame,
        eras: pd.Series,
    ) -> pd.DataFrame:
        self.features = [col for col in dataf.columns if col not in ['era', 'target']]
        print(f"Quantiling for {len(self.features)} features.")
        dataf.loc[:, "era"] = eras
        date_groups = dataf.groupby('era', group_keys=False)
        output_df = pd.DataFrame()
        for feature in tqdm(self.features):
            group_data = date_groups[feature].apply(lambda x: self._process_feature(x))
            output_df[f"{feature}_quantile{self.num_quantiles}"] = group_data
        return output_df
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        if not input_features:
            feature_names = []
            for feature in self.features:
                feature_names.append(f"{feature}_quantile{self.num_quantiles}")
        else:
            feature_names = input_features
        return feature_names


class TickerMapper(BasePreProcessor):
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

    def transform(self, X: Union[np.array, pd.Series]) -> pd.Series:
        """
        Transform ticker column.
        :param X: Ticker column
        :return tickers: Mapped tickers
        """
        tickers = pd.DataFrame(X, columns=[self.ticker_col])[self.ticker_col].map(self.mapping)
        return tickers
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [self.target_ticker_format] if not input_features else input_features
    


class SignalsTargetProcessor(BasePreProcessor):
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

    def transform(self, dataf: pd.DataFrame) -> pd.DataFrame:
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
        output_cols = self.get_feature_names_out()
        return dataf[output_cols]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names of Signals targets. """
        if not input_features:
            feature_names = []
            for window in self.windows:
                feature_names.append(f"target_{window}d_raw")
                feature_names.append(f"target_{window}d_rank")
                feature_names.append(f"target_{window}d_group")
        else:
            feature_names = input_features
        return feature_names


class LagPreProcessor(BasePreProcessor):
    """
    Add lag features based on given windows.

    :param windows: All lag windows to process for all features. \n
    [5, 10, 15, 20] by default (4 weeks lookback) \n
    """

    def __init__(self, windows: list = None,):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]

    def transform(self, X: pd.DataFrame, tickers: pd.Series) -> pd.DataFrame:
        feature_cols = X.columns.tolist()
        X["ticker"] = tickers
        ticker_groups = X.groupby("ticker")
        output_features = []
        for feature in tqdm(feature_cols, desc="Lag feature generation"):
            feature_group = ticker_groups[feature]
            for day in self.windows:
                shifted = feature_group.shift(day, axis=0)
                X.loc[:, f"{feature}_lag{day}"] = shifted
                output_features.append(f"{feature}_lag{day}")
        self.output_features = output_features
        return X[output_features]
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return self.output_features if not input_features else input_features


class DifferencePreProcessor(BasePreProcessor):
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
        pct_diff: bool = False,
        abs_diff: bool = False,
    ):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]
        self.pct_diff = pct_diff
        self.abs_diff = abs_diff

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create difference feature from lag features.
        :param X: DataFrame with lag features.
        NOTE: Make sure only lag features are present in the DataFrame.
        """
        feature_names = X.columns.tolist()
        for col in feature_names:
            assert "_lag" in col, "DifferencePreProcessor expects only lag features. Got feature: '{col}'"
        output_features = []
        for feature in tqdm(feature_names, desc="Difference feature generation"):
            for day in self.windows:
                differenced_values = (
                        (X[feature] / X[feature]) - 1
                        if self.pct_diff
                        else X[feature] - X[feature]
                    )
                X[f"{feature}_diff{day}"] = differenced_values
                output_features.append(f"{feature}_diff{day}")
                if self.abs_diff:
                    X[f"{feature}_absdiff{day}"] = np.abs(
                            X[f"{feature}_diff{day}"]
                        )
                    output_features.append(f"{feature}_absdiff{day}")
        self.output_features = output_features
        return X[self.output_features]
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.output_features if not input_features else input_features


class PandasTaFeatureGenerator(BasePreProcessor):
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


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature generation method. \n 
        :param X: DataFrame with columns: [ticker, date, open, high, low, close, volume] \n
        :return: PandasTA features
        """
        initial_features = X.columns.tolist()
        dataf_list = [
            x
            for _, x in tqdm(
                X.groupby(self.ticker_col), desc="Generating ticker DataFrames"
            )
        ]
        X = self._generate_features(dataf_list=dataf_list)
        output_df = X.drop(columns=initial_features)
        self.output_cols = output_df.columns.tolist()
        return output_df
    
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
        # Run strategy
        ticker_df.ta.strategy(self.strategy)
        return ticker_df
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.output_cols if not input_features else input_features

import os
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from typing import Union, Tuple, List
from multiprocessing.pool import Pool
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import QuantileTransformer

from numerblox.preprocessing.base import BasePreProcessor

# Ignore Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

class ReduceMemoryProcessor(BasePreProcessor):
    """
    Reduce memory usage as much as possible.

    Credits to kainsama and others for writing about memory usage reduction for Numerai data:
    https://forum.numer.ai/t/reducing-memory/313

    :param deep_mem_inspect: Introspect the data deeply by interrogating object dtypes.
    Yields a more accurate representation of memory usage if you have complex object columns.
    :param verbose: Print memory usage before and after optimization.
    """

    def __init__(self, deep_mem_inspect=False, verbose=True):
        super().__init__()
        self.deep_mem_inspect = deep_mem_inspect
        self.verbose = verbose

    def transform(self, dataf: Union[np.array, pd.DataFrame]) -> np.array:
        return self._reduce_mem_usage(dataf).to_numpy()

    def _reduce_mem_usage(self, dataf: Union[np.array, pd.DataFrame]) -> pd.DataFrame:
        """
        Iterate through all columns and modify the numeric column types
        to reduce memory usage.
        """
        dataf = pd.DataFrame(dataf)
        self.output_cols = dataf.columns.tolist()
        start_memory_usage = (
            dataf.memory_usage(deep=self.deep_mem_inspect).sum() / 1024**2
        )
        if self.verbose:
            print(
                f"Memory usage of DataFrame is {round(start_memory_usage, 2)} MB"
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
        if self.verbose:
            print(
                f"Memory usage after optimization is: {round(end_memory_usage, 2)} MB"
            )
            print(
                f"Usage decreased by {round(100 * (start_memory_usage - end_memory_usage) / start_memory_usage, 2)}%"
            )
        return dataf
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return self.output_cols if not input_features else input_features


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
    :param num_cores: Number of cores to use for multiprocessing. \n
    :param verbose: Print additional information.
    """

    warnings.filterwarnings("ignore")
    def __init__(
        self,
        windows: list,
        ticker_col: str = "ticker",
        close_col: str = "close",
        num_cores: int = None,
        verbose=True
    ):
        super().__init__()
        self.windows = windows
        self.ticker_col = ticker_col
        self.close_col = close_col
        self.num_cores = num_cores if num_cores else os.cpu_count()
        self.verbose = verbose

    def transform(self, dataf: pd.DataFrame) -> np.array:
        """
        Multiprocessing feature engineering.
        
        :param dataf: DataFrame with columns: [ticker, date, open, high, low, close, volume] \n
        """
        tickers = dataf.loc[:, self.ticker_col].unique().tolist()
        if self.verbose:
            print(
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
        return dataf[output_cols].to_numpy()

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
        return dataf

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
    """
    Transform features into quantiles by era.
    :param num_quantiles: Number of quantiles to use for quantile transformation. 
    :param random_state: Random state for QuantileTransformer. 
    :param cpu_cores: Number of CPU cores to use for parallel processing.  
    """
    def __init__(
        self,
        num_quantiles: int = 50,
        random_state: int = 0,
        cpu_cores: int = -1,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.random_state = random_state
        self.cpu_cores = cpu_cores
        self.quantiler = QuantileTransformer(
            n_quantiles=self.num_quantiles, random_state=self.random_state
        )

    def _quantile_transform(self, group_data: pd.Series) -> pd.Series:
        """
        Process single feature for a single era.
        :param group_data: Data for a single feature and era.
        :return: Quantile transformed data.
        """
        transformed_data = self.quantiler.fit_transform(group_data.to_frame()).ravel()
        return pd.Series(transformed_data, index=group_data.index)
    
    def fit(self, X: Union[np.array, pd.DataFrame], y=None, eras: pd.Series = None):
        return self

    def transform(
        self, X: Union[np.array, pd.DataFrame],
        eras: pd.Series,
    ) -> np.array:
        """ 
        Quantile all features by era.
        :param X: Array or DataFrame containing features to be quantiled.
        :param eras: Series containing era information.
        :return: Quantiled features.
        """
        X = pd.DataFrame(X)
        assert X.shape[0] == eras.shape[0], "Input X and eras must have the same number of rows for quantiling."
        self.features = [col for col in X.columns]
        X.loc[:, "era"] = eras
        date_groups = X.groupby('era', group_keys=False)

        def process_feature(feature):
            group_data = date_groups[feature].apply(lambda x: self._quantile_transform(x))
            return pd.Series(group_data, name=f"{feature}_quantile{self.num_quantiles}")

        output_series_list = Parallel(n_jobs=self.cpu_cores)(
            delayed(process_feature)(feature) for feature in tqdm(self.features, desc=f"Quantiling {len(self.features)} features")
        )
        output_df = pd.concat(output_series_list, axis=1)
        return output_df.to_numpy()
    
    def fit_transform(self, X: Union[np.array, pd.DataFrame], eras: pd.Series):
        self.fit(X=X, eras=eras)
        return self.transform(X=X, eras=eras)

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

    def transform(self, X: Union[np.array, pd.Series]) -> np.array:
        """
        Transform ticker column.
        :param X: Ticker column
        :return tickers: Mapped tickers
        """
        tickers = pd.DataFrame(X, columns=[self.ticker_col])[self.ticker_col].map(self.mapping)
        return tickers.to_numpy()
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [self.target_ticker_format] if not input_features else input_features
    

class LagPreProcessor(BasePreProcessor):
    """
    Add lag features based on given windows.

    :param windows: All lag windows to process for all features. \n
    [5, 10, 15, 20] by default (4 weeks lookback) \n
    """

    def __init__(self, windows: list = None,):
        super().__init__()
        self.windows = windows if windows else [5, 10, 15, 20]

    def fit(self, X: Union[np.array, pd.DataFrame], y=None, tickers: pd.Series = None):
        return self

    def transform(self, X: Union[np.array, pd.DataFrame], tickers: pd.Series) -> np.array:
        X = pd.DataFrame(X)
        feature_cols = X.columns.tolist()
        X["ticker"] = tickers
        ticker_groups = X.groupby("ticker")
        output_features = []
        for feature in tqdm(feature_cols, desc="Lag feature generation"):
            feature_group = ticker_groups[feature]
            for day in self.windows:
                shifted = feature_group.shift(day)
                X.loc[:, f"{feature}_lag{day}"] = shifted
                output_features.append(f"{feature}_lag{day}")
        self.output_features = output_features
        return X[output_features].to_numpy()
    
    def fit_transform(self, X: Union[np.array, pd.DataFrame], tickers: pd.Series):
        self.fit(X=X, tickers=tickers)
        return self.transform(X=X, tickers=tickers)
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return self.output_features if not input_features else input_features


class DifferencePreProcessor(BasePreProcessor):
    """
    Add difference features based on given windows. Run LagPreProcessor first.
    Usage in Pipeline works only with Pandas API. 
    Run `.set_output("pandas")` on your pipeline first.

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

    def transform(self, X: pd.DataFrame) -> np.array:
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
        return X[self.output_features].to_numpy()
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.output_features if not input_features else input_features


class PandasTaFeatureGenerator(BasePreProcessor):
    """
    Generate features with pandas-ta.
    https://github.com/twopirllc/pandas-ta
    Usage in Pipeline works only with Pandas API. 
    Run `.set_output("pandas")` on your pipeline first.

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


class HLOCVAdjuster(BasePreProcessor):
    """ 
    Adjust HLOCV data for splits and dividends based on ratio of unadjusted and adjusted close prices.
    NOTE: This step only works with DataFrame input. 
    Usage in intermediate steps of a scikit-learn Pipeline works with the Pandas set_output API.
    i.e. pipeline.set_output(transform="pandas").
    """
    def __init__(self, open_col="open", high_col="high", low_col="low", 
                 close_col="close", volume_col="volume", adj_close_col="adjusted_close"):
        super().__init__()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.adj_close_col = adj_close_col
        self.adjusted_col_names = [f"adjusted_{self.high_col}", f"adjusted_{self.low_col}",
                                   f"adjusted_{self.open_col}", self.adj_close_col, 
                                   f"adjusted_{self.volume_col}"]

    def fit(self, X: pd.DataFrame, y=None):
        self.ratio_ = X[self.close_col] / X[self.adj_close_col]
        return self

    def transform(self, X: pd.DataFrame) -> np.array:
        """
        Adjust open, high, low, close and volume for splits and dividends.
        :param X: DataFrame with columns: [high, low, open, close, volume] (HLOCV)
        :return: Array with adjusted HLOCV columns
        """
        X_copy = X.copy()  
        X_copy[f"adjusted_{self.high_col}"] = X[self.high_col] / self.ratio_
        X_copy[f"adjusted_{self.low_col}"] = X[self.low_col] / self.ratio_
        X_copy[f"adjusted_{self.open_col}"] = X[self.open_col] / self.ratio_
        X_copy[f"adjusted_{self.volume_col}"] = X[self.volume_col] * self.ratio_
        return X_copy[self.adjusted_col_names].to_numpy()
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.adjusted_col_names


class MinimumDataFilter(BasePreProcessor):
    """ 
    Filter dates and tickers based on minimum data requirements. 
    NOTE: This step only works with DataFrame input.

    :param min_samples_date: Minimum number of samples per date. Defaults to 200.
    :param min_samples_ticker: Minimum number of samples per ticker. Defaults to 1200.
    :param blacklist_tickers: List of tickers to exclude from the dataset. Defaults to None.
    :param date_col: Column name for date. Defaults to "date".
    :param ticker_col: Column name for ticker. Defaults to "bloomberg_ticker".
    """
    def __init__(self, min_samples_date: int = 200, min_samples_ticker: int = 1200, blacklist_tickers: list = None, date_col="date", ticker_col="bloomberg_ticker"):
        super().__init__()
        self.min_samples_date = min_samples_date
        self.min_samples_ticker = min_samples_ticker
        self.blacklist_tickers = blacklist_tickers
        self.date_col = date_col
        self.ticker_col = ticker_col

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.array:
        """
        Filter dates and tickers based on minimum data requirements.
        :param X: DataFrame with columns: [ticker_col, date_col, open, high, low, close, volume] (HLOCV)
        :return: Array with filtered DataFrame
        """
        filtered_data = X.groupby(self.date_col).filter(lambda x: len(x) >= self.min_samples_date)
        records_per_ticker = (
            filtered_data.reset_index(drop=False)
            .groupby(self.ticker_col)[self.date_col]
            .nunique()
            .reset_index()
            .sort_values(by=self.date_col)
        )
        tickers_with_records = records_per_ticker.query(f"{self.date_col} >= {self.min_samples_ticker}")[self.ticker_col].values
        filtered_data = filtered_data.loc[filtered_data[self.ticker_col].isin(tickers_with_records)].reset_index(drop=True)

        if self.blacklist_tickers:
            filtered_data = filtered_data.loc[~filtered_data[self.ticker_col].isin(self.blacklist_tickers)]

        return filtered_data.to_numpy()
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        check_is_fitted(self)
        return self.feature_names_out_ if not input_features else input_features
    
import warnings
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from numerblox.preprocessing.base import BasePreProcessor
from numerblox.preprocessing import (ReduceMemoryProcessor, GroupStatsPreProcessor,
                                     KatsuFeatureGenerator,
                                     EraQuantileProcessor, TickerMapper,
                                     LagPreProcessor, 
                                     DifferencePreProcessor, PandasTaFeatureGenerator, HLOCVAdjuster,
                                     MinimumDataFilter)
from numerblox.feature_groups import V4_2_FEATURE_GROUP_MAPPING

from utils import create_signals_sample_data

CLASSIC_PREPROCESSORS = [ReduceMemoryProcessor, GroupStatsPreProcessor]
SIGNALS_PREPROCESSORS = [KatsuFeatureGenerator, EraQuantileProcessor, TickerMapper,
                         LagPreProcessor, DifferencePreProcessor, PandasTaFeatureGenerator, HLOCVAdjuster,
                         MinimumDataFilter]
ALL_PREPROCESSORS = CLASSIC_PREPROCESSORS + SIGNALS_PREPROCESSORS
WINDOW_COL_PROCESSORS = [KatsuFeatureGenerator, LagPreProcessor, 
                         DifferencePreProcessor]
TICKER_PROCESSORS = [LagPreProcessor]

dataset = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")
dummy_signals_data = create_signals_sample_data

def test_base_preprocessor():
    assert hasattr(BasePreProcessor, 'fit')
    assert hasattr(BasePreProcessor, 'transform')
    assert issubclass(BasePreProcessor, (BaseEstimator, TransformerMixin))

def test_processors_sklearn(dummy_signals_data):
    data = dataset.sample(50)
    data = data.drop(columns=["data_type"])
    data['ticker'] = ["AAPL US"] * 25 + ["MSFT US"] * 25
    y = data["target_jerome_v4_20"].fillna(0.5)
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    classic_X = data[feature_names].fillna(0.5)
    signals_X = dummy_signals_data[["open", "high", "low", "close", "volume", "adjusted_close"]]

    for processor_cls in tqdm(ALL_PREPROCESSORS, desc="Testing processors for scikit-learn compatibility"):
        # Initialization
        if processor_cls in WINDOW_COL_PROCESSORS:
            processor = processor_cls(windows=[20, 40])
        else:
            processor = processor_cls()

        if processor_cls in CLASSIC_PREPROCESSORS:
            X = classic_X
        elif processor_cls in SIGNALS_PREPROCESSORS:
            X = signals_X

        # Test fit returns self
        assert processor.fit(X=X, y=y) == processor

        # Inherits from BasePreProcessor
        assert issubclass(processor_cls, BasePreProcessor)
        # Has fit_transform
        assert hasattr(processor_cls, 'fit_transform')

        # Pipeline
        pipeline = Pipeline([
                ('processor', processor)
            ])
        _ = pipeline.fit(X)

        # FeatureUnion
        combined_features = FeatureUnion([
                ('processor', processor),
                ('pca', PCA())
            ])
        _ = combined_features.fit(X)

        # Test every processor has get_feature_names_out
        assert hasattr(processor, 'get_feature_names_out'), "Processor {processor.__name__} does not have get_feature_names_out. Every implemented preprocessor should have this method."

def test_reduce_memory_preprocessor(dummy_signals_data):
    # Reduce memory
    rmp = ReduceMemoryProcessor()
    rmp.set_output(transform="pandas")
    reduced_data = rmp.fit_transform(dummy_signals_data)
    # Check types
    assert reduced_data.close.dtype == "O"
    assert reduced_data.volume.dtype == "O"
    assert reduced_data.date.dtype == "<M8[ns]"
    assert rmp.get_feature_names_out() == dummy_signals_data.columns.tolist()

    # Test numpy input and set_output API
    rmp.set_output(transform="default")
    reduced_data = rmp.transform(dummy_signals_data.to_numpy())
    assert isinstance(reduced_data, np.ndarray)

    # Test polars Output
    rmp.set_output(transform="polars")
    reduced_data = rmp.transform(dummy_signals_data)
    assert isinstance(reduced_data, pl.DataFrame)


def test_group_stats_preprocessor():
    # Test with part groups selects
    test_group_processor = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    test_group_processor.set_output(transform="pandas")
    assert test_group_processor.group_names == ["sunshine", "rain"]

    result = test_group_processor.fit_transform(dataset)

    expected_cols = [
        "feature_sunshine_mean", "feature_sunshine_std", "feature_sunshine_skew",
        "feature_rain_mean", "feature_rain_std", "feature_rain_skew"
    ]
    for col in expected_cols:
        assert col in result.columns
        # Mean should be between 0 and 4
        if col.endswith("mean"):
            assert result[col].min() >= 0.0
            assert result[col].max() <= 4.0
        # Std should be between 0 and 2
        if col.endswith("std"):
            assert result[col].min() >= 0.0
            assert result[col].max() <= 2.0

    random_rain_features = np.random.choice(V4_2_FEATURE_GROUP_MAPPING['rain'], size=10).tolist()
    # Warn if not all columns of a group are in the dataset
    processor = GroupStatsPreProcessor(groups=['rain'])
    processor.set_output(transform="pandas")
    with warnings.catch_warnings(record=True) as w:
        result = processor.transform(dataset[random_rain_features])
        assert issubclass(w[-1].category, UserWarning)
        assert f"Not all columns of 'rain' are in the input data" in str(w[-1].message)
        # Check output has no nans
        assert not result.isna().any().any()
        # Check Mean between 0 and 4
        assert result["feature_rain_mean"].min() >= 0.0
        assert result["feature_rain_mean"].max() <= 4.0
        # Check Std between 0 and 2
        assert result["feature_rain_std"].min() >= 0.0
        assert result["feature_rain_std"].max() <= 2.0

    # Warn if none of the columns of a group are in the dataset
    processor = GroupStatsPreProcessor(groups=['intelligence'])
    processor.set_output(transform="pandas")
    with warnings.catch_warnings(record=True) as w:
        result = processor.transform(dataset[random_rain_features])
        assert issubclass(w[-1].category, UserWarning)
        assert "None of the columns of 'intelligence' are in the input data. Output will be nans for the group features." in str(w[-1].message)
        # Check result contains only nans
        assert result.isna().all().all()
    
    # Test set_output API
    processor.set_output(transform="default")
    result = processor.transform(dataset)
    assert isinstance(result, np.ndarray)

    processor.set_ouput(transform="polars")
    result = processor.transform(dataset)
    assert isinstance(result, pl.DataFrame)

    # Test get_feature_names_out
    assert test_group_processor.get_feature_names_out() == expected_cols
    assert test_group_processor.get_feature_names_out(["fancy"]) == ["fancy"]

def test_katsu_feature_generator(dummy_signals_data):
    kfg = KatsuFeatureGenerator(windows=[20, 40])
    kfg.set_output(transform="pandas")
    result = kfg.fit_transform(dummy_signals_data)
    expected_cols = [
    "feature_close_ROCP_20", 
    "feature_close_VOL_20",
    "feature_close_MA_gap_20",
    "feature_close_ROCP_40",
    "feature_close_VOL_40",
    "feature_close_MA_gap_40",
    "feature_RSI",
    "feature_MACD",
    "feature_MACD_signal"
]
    assert result.columns.tolist() == expected_cols
    assert kfg.get_feature_names_out() == expected_cols

    # Test set_output API
    kfg.set_output(transform="default")
    result = kfg.transform(dummy_signals_data)
    assert isinstance(result, np.ndarray)

def test_era_quantile_processor(dummy_signals_data):
    eqp = EraQuantileProcessor(num_quantiles=2)
    eqp.set_output(transform="pandas")
    X = dummy_signals_data[["close", "volume"]]
    eqp.fit(X)
    result = eqp.transform(X, eras=dummy_signals_data["date"])
    quantile_cols = [col for col in result.columns if "quantile" in col]
    assert len(result.columns) == 2
    for col in quantile_cols:
        assert result[col].min() >= 0.0
        assert result[col].max() <= 1.0
    assert eqp.get_feature_names_out() == quantile_cols

    # Numpy input
    result = eqp.transform(X.to_numpy(), eras=dummy_signals_data["date"])
    assert len(result.shape) == 2
    assert isinstance(result, pd.DataFrame)

    # Test set_output API
    eqp.set_output(transform="default")
    result = eqp.transform(X, eras=dummy_signals_data["date"])
    assert isinstance(result, np.ndarray)

    eqp.set_output(transform="polars")
    result = eqp.transform(X, eras=dummy_signals_data["date"])
    assert isinstance(result, pl.DataFrame)

def test_ticker_mapper():
    # Basic
    test_dataf = pd.Series(["AAPL", "MSFT"])
    mapper = TickerMapper()
    result = mapper.fit_transform(test_dataf)
    assert result.tolist() == ["AAPL US", "MSFT US"]

    # From CSV
    test_dataf = pd.Series(["LLB SW", "DRAK NA", "SWB MK", "ELEKTRA* MF", "NOT_A_TICKER"])
    mapper = TickerMapper(ticker_col="bloomberg_ticker", target_ticker_format="signals_ticker",
                        mapper_path="tests/test_assets/eodhd-map.csv")
    result = mapper.transform(test_dataf)
    assert result.tolist() == ["LLB.SW", "DRAK.AS", "5211.KLSE", "ELEKTRA.MX", np.nan]

    # Test set_output API
    mapper.set_output(transform="default")
    result = mapper.transform(test_dataf)
    assert isinstance(result, np.ndarray)

    mapper.set_output(transform="polars")
    result = mapper.transform(test_dataf)
    assert isinstance(result, pl.Series)
    

def test_lag_preprocessor(dummy_signals_data):
    lpp = LagPreProcessor(windows=[20, 40])
    lpp.set_output(transform="pandas")
    lpp.fit(dummy_signals_data[['close', 'volume']])
    # DataFrame input
    result = lpp.transform(dummy_signals_data[['close', 'volume']], tickers=dummy_signals_data["ticker"])
    expected_cols = [
    "close_lag20",
    "close_lag40",
    "volume_lag20",
    "volume_lag40",
]
    assert result.columns.tolist() == expected_cols
    assert lpp.get_feature_names_out() == expected_cols

    # Numpy input
    result = lpp.transform(dummy_signals_data[['close', 'volume']].to_numpy(), tickers=dummy_signals_data["ticker"])
    expected_cols = [
    "0_lag20",
    "0lag40",
    "1_lag20",
    "1_lag40",
]

    # Test set_output API
    lpp.set_output(transform="default")
    result = lpp.transform(dummy_signals_data[['close', 'volume']], tickers=dummy_signals_data["ticker"])
    assert isinstance(result, np.ndarray)

    lpp.set_output(transform="polars")
    result = lpp.transform(dummy_signals_data[['close', 'volume']], tickers=dummy_signals_data["ticker"])
    assert isinstance(result, pl.DataFrame)


def test_difference_preprocessor(dummy_signals_data):
    lpp = LagPreProcessor(windows=[20, 40])
    lpp.set_output(transform="pandas")
    lpp.fit(dummy_signals_data[['close', 'volume']])
    lags = lpp.transform(dummy_signals_data[['close', 'volume']],
                         tickers=dummy_signals_data["ticker"])
    dpp = DifferencePreProcessor(windows=[20, 40], abs_diff=True)
    dpp.set_output(transform="pandas")
    result = dpp.fit_transform(lags)
    assert result.columns.tolist() == ['close_lag20_diff20', 'close_lag20_absdiff20', 'close_lag20_diff40', 'close_lag20_absdiff40', 'close_lag40_diff20', 'close_lag40_absdiff20', 'close_lag40_diff40', 'close_lag40_absdiff40', 'volume_lag20_diff20', 'volume_lag20_absdiff20', 'volume_lag20_diff40',
    'volume_lag20_absdiff40', 'volume_lag40_diff20',
    'volume_lag40_absdiff20', 'volume_lag40_diff40',
    'volume_lag40_absdiff40']

    # Test set_output API
    dpp.set_output(transform="default")
    result = dpp.transform(lags)
    assert isinstance(result, np.ndarray)

    dpp.set_output(transform="polars")
    result = dpp.transform(lags)
    assert isinstance(result, pl.DataFrame)

def test_pandasta_feature_generator(dummy_signals_data):
    ptfg = PandasTaFeatureGenerator()
    result = ptfg.fit_transform(dummy_signals_data)
    expected_cols = ["feature_RSI_14", "feature_RSI_60"]
    assert result.columns.tolist() == expected_cols
    assert ptfg.get_feature_names_out() == expected_cols

def test_hlocv_adjuster_basic(dummy_signals_data):
    adjuster = HLOCVAdjuster()
    adjuster.fit(dummy_signals_data)
    
    adjusted_array = adjuster.transform(dummy_signals_data)
    adjusted_df = pd.DataFrame(adjusted_array, columns=adjuster.get_feature_names_out())
    
    # Check for column presence
    assert all(col in adjusted_df.columns for col in adjuster.get_feature_names_out())

    # Check if the ratio is maintained correctly. 
    original_row = dummy_signals_data.iloc[0]
    adjusted_row = adjusted_df.iloc[0]

    ratio = original_row["close"] / original_row["adjusted_close"]
    assert np.isclose(original_row["high"] / ratio, adjusted_row["adjusted_high"])
    assert np.isclose(original_row["low"] / ratio, adjusted_row["adjusted_low"])
    assert np.isclose(original_row["open"] / ratio, adjusted_row["adjusted_open"])
    assert np.isclose(original_row["volume"] * ratio, adjusted_row["adjusted_volume"])

    # Test set_output API
    adjuster.set_output(transform="default")
    result = adjuster.transform(dummy_signals_data)
    assert isinstance(result, np.ndarray)

    adjuster.set_output(transform="polars")
    result = adjuster.transform(dummy_signals_data)
    assert isinstance(result, pl.DataFrame)

def test_minimum_data_filter(dummy_signals_data):
    before_tickers = dummy_signals_data["ticker"].unique().tolist()
    for tick in ["XYZ.US", "RST.US", "UVW.US"]:
        assert tick in before_tickers
    filter = MinimumDataFilter(ticker_col="ticker", date_col="date", min_samples_date=2, min_samples_ticker=50)
    filter.fit(dummy_signals_data)
    filtered_data = pd.DataFrame(filter.transform(dummy_signals_data), columns=filter.get_feature_names_out())
    # Some tickers should have been filtered (XYZ.US, RST.US, UVW.US)
    assert not filtered_data.empty
    assert filtered_data.shape[0] < dummy_signals_data.shape[0]
    assert len(filtered_data["ticker"].unique()) < len(dummy_signals_data["ticker"].unique())
    after_tickers = filtered_data["ticker"].unique().tolist()
    for tick in ["XYZ.US", "RST.US", "UVW.US"]:
        assert tick not in after_tickers
    assert filtered_data.shape[1] == dummy_signals_data.shape[1]
    assert filter.get_feature_names_out() == dummy_signals_data.columns.tolist()

    # Test set_output API
    filter.set_output(transform="default")
    result = filter.transform(dummy_signals_data)
    assert isinstance(result, np.ndarray)

    filter.set_output(transform="polars")
    result = filter.transform(dummy_signals_data)
    assert isinstance(result, pl.DataFrame)


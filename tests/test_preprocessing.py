import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


from numerblox.preprocessing import (BasePreProcessor,
                                     ReduceMemoryProcessor, GroupStatsPreProcessor, KatsuFeatureGenerator,
                                     EraQuantileProcessor, TickerMapper,
                                     LagPreProcessor, 
                                     DifferencePreProcessor, PandasTaFeatureGenerator, V4_2_FEATURE_GROUP_MAPPING)

from utils import create_signals_sample_data, create_classic_sample_data

CLASSIC_PREPROCESSORS = [ReduceMemoryProcessor, GroupStatsPreProcessor]
SIGNALS_PREPROCESSORS = [KatsuFeatureGenerator, EraQuantileProcessor, TickerMapper,
                         LagPreProcessor, DifferencePreProcessor, PandasTaFeatureGenerator]
ALL_PREPROCESSORS = CLASSIC_PREPROCESSORS + SIGNALS_PREPROCESSORS
WINDOW_COL_PROCESSORS = [KatsuFeatureGenerator, LagPreProcessor, 
                         DifferencePreProcessor]

dataset = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")
dummy_classic_data = create_classic_sample_data
dummy_signals_data = create_signals_sample_data

def test_base_preprocessor():
    assert hasattr(BasePreProcessor, 'fit')
    assert hasattr(BasePreProcessor, 'transform')
    assert issubclass(BasePreProcessor, (BaseEstimator, TransformerMixin))


def test_processors_sklearn():
    data = dataset.sample(50)
    data = data.drop(columns=["data_type"])
    y = data["target_jerome_v4_20"].fillna(0.5)
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    X = data[feature_names]

    for processor_cls in tqdm(ALL_PREPROCESSORS, desc="Testing processors for scikit-learn compatibility"):
        # Initialization
        if processor_cls in WINDOW_COL_PROCESSORS:
            processor = processor_cls(windows=[20, 40])
        else:
            processor = processor_cls()

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
        _ = pipeline.fit(data)

        # FeatureUnion
        combined_features = FeatureUnion([
                ('processor', processor),
                ('pca', PCA())
            ])
        _ = combined_features.fit(data.fillna(0.5))
        # TODO Test with NumeraiPipeline

        # Test every processor has get_feature_names_out
        assert hasattr(processor, 'get_feature_names_out'), "Processor {processor.__name__} does not have get_feature_names_out. Every implemented preprocessors should have this method."

def test_reduce_memory_preprocessor(dummy_classic_data):
    # Reduce memory
    rmp = ReduceMemoryProcessor()
    reduced_data = rmp.fit_transform(dummy_classic_data)
    # Check types
    assert reduced_data.feature1.dtype == "int16"
    assert reduced_data.feature2.dtype == "int16"
    assert reduced_data.era.dtype == "O"
    assert rmp.get_feature_names_out() == dummy_classic_data.columns.tolist()


def test_group_stats_preprocessor():

    # Test with part groups selects
    test_group_processor = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    assert test_group_processor.group_names == ["sunshine", "rain"]

    result = GroupStatsPreProcessor().fit_transform(dataset)

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
    with warnings.catch_warnings(record=True) as w:
        result = processor.transform(dataset[random_rain_features])
        assert issubclass(w[-1].category, UserWarning)
        assert "None of the columns of 'intelligence' are in the input data. Output will be nans for the group features." in str(w[-1].message)
        # Check result contains only nans
        assert result.isna().all().all()

    # Test get_feature_names_out
    assert test_group_processor.get_feature_names_out() == expected_cols




def test_katsu_feature_generator(dummy_signals_data):
    kfg = KatsuFeatureGenerator(windows=[20, 40])
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


def test_era_quantile_processor(dummy_signals_data):
    eqp = EraQuantileProcessor(num_quantiles=2)
    X = dummy_signals_data[["close", "volume"]]
    eqp.fit(X)
    result = eqp.transform(X, eras=dummy_signals_data["date"])
    quantile_cols = [col for col in result.columns if "quantile" in col]
    assert len(result.columns) == 2
    for col in quantile_cols:
        assert result[col].min() >= 0.0
        assert result[col].max() <= 1.0
    assert eqp.get_feature_names_out() == quantile_cols

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

    
def test_lag_preprocessor(dummy_signals_data):
    lpp = LagPreProcessor(windows=[20, 40])
    lpp.fit(dummy_signals_data[['close', 'volume']])
    result = lpp.transform(dummy_signals_data[['close', 'volume']], tickers=dummy_signals_data["ticker"])
    expected_cols = [
    "close_lag20",
    "close_lag40",
    "volume_lag20",
    "volume_lag40",
]
    assert result.columns.tolist() == expected_cols
    assert lpp.get_feature_names_out() == expected_cols

def test_difference_preprocessor(dummy_signals_data):
    lpp = LagPreProcessor(windows=[20, 40])
    lpp.fit(dummy_signals_data[['close', 'volume']])
    lags = lpp.transform(dummy_signals_data[['close', 'volume']],
                         tickers=dummy_signals_data["ticker"])
    dpp = DifferencePreProcessor(windows=[20, 40], abs_diff=True)
    result = dpp.fit_transform(lags)
    assert result.columns.tolist() == ['close_lag20_diff20', 'close_lag20_absdiff20', 'close_lag20_diff40', 'close_lag20_absdiff40', 'close_lag40_diff20', 'close_lag40_absdiff20', 'close_lag40_diff40', 'close_lag40_absdiff40', 'volume_lag20_diff20', 'volume_lag20_absdiff20', 'volume_lag20_diff40',
    'volume_lag20_absdiff40', 'volume_lag40_diff20',
    'volume_lag40_absdiff20', 'volume_lag40_diff40',
    'volume_lag40_absdiff40']

def test_pandasta_feature_generator(dummy_signals_data):
    ptfg = PandasTaFeatureGenerator()
    result = ptfg.fit_transform(dummy_signals_data)
    expected_cols = ["feature_RSI_14", "feature_RSI_60"]
    assert result.columns.tolist() == expected_cols
    assert ptfg.get_feature_names_out() == expected_cols

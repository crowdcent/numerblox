import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


from numerblox.numerframe import NumerFrame, create_numerframe
from numerblox.preprocessing import (BasePreProcessor, CopyPreProcessor,
                                     FeatureSelectionPreProcessor, TargetSelectionPreProcessor,
                                     ReduceMemoryProcessor, GroupStatsPreProcessor, KatsuFeatureGenerator,
                                     EraQuantileProcessor, TickerMapper, SignalsTargetProcessor, LagPreProcessor, 
                                     DifferencePreProcessor, PandasTaFeatureGenerator)

from utils import create_signals_sample_data

CLASSIC_PREPROCESSORS = ['CopyPreProcessor', 'FeatureSelectionPreProcessor',
                     'TargetSelectionPreProcessor', 'ReduceMemoryProcessor',
                     'BayesianGMMTargetProcessor', 
                     'GroupStatsPreProcessor']
SIGNALS_PREPROCESSORS = ['KatsuFeatureGenerator', 'EraQuantileProcessor',
                     'TickerMapper', 'SignalsTargetProcessor', 'LagPreProcessor', 'DifferencePreProcessor', 'PandasTaFeatureGenerator', 'AwesomePreProcessor']
ALL_PREPROCESSORS = CLASSIC_PREPROCESSORS + SIGNALS_PREPROCESSORS
FEATURE_COL_PROCESSORS = ["FeatureSelectionPreProcessor"]
TARGET_COL_PROCESSORS = ["TargetSelectionPreProcessor"]
WINDOW_COL_PROCESSORS = ["KatsuFeatureGenerator", "LagPreProcessor", "DifferencePreProcessor"]


MODULE = "numerblox.preprocessing"
module = importlib.import_module(MODULE)
processors = [getattr(module, proc_name) for proc_name in ALL_PREPROCESSORS if hasattr(module, proc_name)]
dataset = create_numerframe("tests/test_assets/train_int8_5_eras.parquet")

def test_base_preprocessor():
    assert hasattr(BasePreProcessor, 'fit')
    assert hasattr(BasePreProcessor, 'transform')
    assert issubclass(BasePreProcessor, (BaseEstimator, TransformerMixin))


def test_processors_sklearn():
    data = NumerFrame(dataset.copy().loc[:, dataset.feature_cols + dataset.target_cols].sample(10))
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    target_names = ["target_jerome_v4_20"]

    for processor_cls in tqdm(processors, desc="Testing processors for scikit-learn compatability"):
        # Initialization
        if processor_cls.__name__ in FEATURE_COL_PROCESSORS:
            processor = processor_cls(feature_cols=feature_names)
        elif processor_cls.__name__ in TARGET_COL_PROCESSORS:
            processor = processor_cls(target_cols=target_names)
        elif processor_cls.__name__ in WINDOW_COL_PROCESSORS:
            processor = processor_cls(windows=[20, 40])
        else:
            processor = processor_cls()

        # Test fit returns self
        assert processor.fit(X=pd.DataFrame()) == processor

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

def test_copypreprocessor():
    data = dataset.copy()
    copied_dataset = CopyPreProcessor().transform(dataset)
    assert copied_dataset.equals(dataset)
    assert dataset.meta == copied_dataset.meta

def test_feature_selection_preprocessor():
    data = dataset.copy()
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    feature_selection_dataset = FeatureSelectionPreProcessor(feature_cols=feature_names).fit_transform(data)
    assert feature_selection_dataset.get_feature_data.columns.tolist() == feature_names
    assert feature_selection_dataset.target_cols == data.target_cols
    assert feature_selection_dataset.meta == data.meta
    assert feature_selection_dataset.get_feature_data.shape[1] == 2

def test_target_selection_preprocessor():
    data = dataset.copy()
    target_names = ["target_jerome_v4_20", "target_william_v4_20"]
    target_selection_dataset = TargetSelectionPreProcessor(target_cols=target_names).fit_transform(data)
    assert target_selection_dataset.get_target_data.columns.tolist() == target_names
    assert target_selection_dataset.feature_cols == data.feature_cols
    assert target_selection_dataset.meta == data.meta
    assert target_selection_dataset.get_target_data.shape[1] == 2

def test_reduce_memory_preprocessor():
    # Random data
    data = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "col3": ["a", "b", "c", "d", "e"],
    })
    # Reduce memory
    reduced_data = ReduceMemoryProcessor().fit_transform(data)
    # Check types
    assert reduced_data.col1.dtype == "int16"
    assert reduced_data.col2.dtype == "float16"
    assert reduced_data.col3.dtype == "O"

# TODO Proper BayesianGMMTargetProcessor test
def test_bayesian_gmm_target_preprocessor():
    pass

def test_group_stats_preprocessor():
    data = dataset.copy()

    # Test with part groups selects
    test_group_dataset = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    assert test_group_dataset.group_names == ["sunshine", "rain"]

    result = GroupStatsPreProcessor().fit_transform(data)
    assert "feature_intelligence_mean" in result.columns
    assert "feature_intelligence_std" in result.columns
    assert "feature_intelligence_skew" in result.columns
    assert "feature_charisma_mean" in result.columns
    assert "feature_charisma_std" in result.columns
    assert "feature_charisma_skew" in result.columns

def test_katsu_feature_generator():
    data = create_signals_sample_data()
    result = KatsuFeatureGenerator(windows=[20, 40]).fit_transform(data)
    assert result.feature_cols == [
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

def test_era_quantile_processor():
    data = create_signals_sample_data()
    result = EraQuantileProcessor(features=['close', 'volume'],
                                  num_quantiles=2).fit_transform(data)
    quantile_cols = [col for col in result.columns if "quantile" in col]
    assert len(quantile_cols) == 2
    for col in quantile_cols:
        assert result[col].min() >= 0.0
        assert result[col].max() <= 1.0

def test_ticker_mapper():
    # Basic
    test_dataf = pd.DataFrame(["AAPL", "MSFT"], columns=["ticker"])
    mapper = TickerMapper()
    result = mapper.fit_transform(test_dataf)
    assert result['bloomberg_ticker'].tolist() == ["AAPL US", "MSFT US"]

    # From CSV
    test_dataf = pd.DataFrame(["LLB SW", "DRAK NA", "SWB MK", "ELEKTRA* MF", "NOT_A_TICKER"], columns=["bloomberg_ticker"])
    mapper = TickerMapper(ticker_col="bloomberg_ticker", target_ticker_format="signals_ticker",
                        mapper_path="tests/test_assets/eodhd-map.csv")
    result = mapper.transform(test_dataf)
    assert result['signals_ticker'].tolist() == ["LLB.SW", "DRAK.AS", "5211.KLSE", "ELEKTRA.MX", np.nan]

def test_signals_target_processor():
    data = create_signals_sample_data()
    result = SignalsTargetProcessor().fit_transform(data)
    expected_target_cols = ["target", "target_10d_raw", "target_10d_rank", "target_10d_group",
                                  "target_20d_raw", "target_20d_rank", "target_20d_group"]
    for col in expected_target_cols:
        assert col in result.columns
    
def test_lag_preprocessor():
    data = create_signals_sample_data()
    result = LagPreProcessor(feature_names=["close", "volume"], windows=[20, 40],
                             ticker_col="ticker").fit_transform(data)
    assert list(result.get_pattern_data("lag").columns) == [
    "close_lag20", 
    "close_lag40",
    "volume_lag20",
    "volume_lag40",
]

def test_difference_preprocessor():
    data = create_signals_sample_data()
    lags = LagPreProcessor(feature_names=["close", "volume"], windows=[20, 40],
                             ticker_col="ticker").fit_transform(data)
    result = DifferencePreProcessor(feature_names=["close", "volume"], windows=[20, 40]).fit_transform(lags)
    assert list(result.get_pattern_data("diff").columns) == [
    "close_diff20",
    "close_diff40",
    "volume_diff20",
    "volume_diff40",
]

def test_pandasta_feature_generator():
    data = create_signals_sample_data()
    result = PandasTaFeatureGenerator().fit_transform(data)
    assert result.feature_cols == [
        "feature_RSI_14",
        "feature_RSI_60"
    ]

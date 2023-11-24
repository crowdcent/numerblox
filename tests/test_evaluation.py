import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from numerblox.misc import Key
from numerblox.evaluation import NumeraiClassicEvaluator, NumeraiSignalsEvaluator

from utils import create_signals_sample_data, classic_test_data

BASE_STATS_COLS = ["target", "mean", "std", "sharpe", 
                   "max_drawdown", "apy", "calmar_ratio", "autocorrelation",
                   "legacy_mean", "legacy_std", "legacy_sharpe"]
CLASSIC_SPECIFIC_STATS_COLS = ["feature_neutral_mean_v3", "feature_neutral_std_v3", 
                               "feature_neutral_sharpe_v3"]


CLASSIC_STATS_COLS = BASE_STATS_COLS + CLASSIC_SPECIFIC_STATS_COLS
SIGNALS_STATS_COLS = BASE_STATS_COLS


def test_numerai_classic_evaluator(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random"] = np.random.uniform(size=len(df))

    evaluator = NumeraiClassicEvaluator(era_col="era", fast_mode=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
        example_col="prediction_random",
    )
    for col in CLASSIC_STATS_COLS + CLASSIC_SPECIFIC_STATS_COLS:
        assert col in val_stats.columns
        assert val_stats[col][0] != np.nan


def test_evaluation_benchmark_cols(classic_test_data):
    df = classic_test_data
    np.random.seed(1)
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random"] = np.random.uniform(size=len(df))
    df.loc[:, "benchmark1"] = np.random.uniform(size=len(df))
    df.loc[:, "benchmark2"] = np.random.uniform(size=len(df))
    benchmark_cols = ["benchmark1", "benchmark2"]

    evaluator = NumeraiClassicEvaluator(era_col="era", fast_mode=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
        benchmark_cols=benchmark_cols,
    )
    additional_expected_cols = []
    for col in benchmark_cols:
        additional_expected_cols.extend([f"corr_with_{col}", 
                                        f"mean_outperformance_vs_{col}", 
                                        f"sharpe_outperformance_vs_{col}",
                                        f"smart_sharpe_outperformance_vs_{col}",
                                        f"legacy_bmc_{col}_mean",
                                        f"legacy_bmc_{col}_std",
                                        f"legacy_bmc_{col}_sharpe",
                                        f"legacy_bmc_{col}_plus_corr_sharpe"])
    for col in CLASSIC_STATS_COLS + CLASSIC_SPECIFIC_STATS_COLS + additional_expected_cols:
        assert col in val_stats.columns
        assert val_stats[col][0] != np.nan


def test_numerai_signals_evaluator(create_signals_sample_data):
    df = create_signals_sample_data
    evaluator = NumeraiSignalsEvaluator(era_col="date", fast_mode=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
        example_col="prediction_random",
    )
    for col in SIGNALS_STATS_COLS:
        assert col in val_stats.columns
        assert val_stats[col][0] != np.nan


@pytest.fixture
def mock_api():
    with patch("numerblox.evaluation.SignalsAPI") as mock:
        mock_instance = mock.return_value
        
        # Mock get_models method
        mock_instance.get_models.return_value = {'test_model': 'test_model_id'}
        
        # Mock upload_diagnostics method
        mock_instance.upload_diagnostics.return_value = "test_diag_id"
        
        # Mock diagnostics method
        mock_instance.diagnostics.return_value = [{"status": "done", "perEraDiagnostics": [{"era": "2023-09-01", "validationCorr": 0.6}]}]
        yield mock_instance


def test_get_neutralized_corr(create_signals_sample_data, mock_api):
    df = create_signals_sample_data
    obj = NumeraiSignalsEvaluator(era_col="date", fast_mode=True)  
    result = obj.get_neutralized_corr(df, "test_model", Key("Hello", "World"))
    
    # Asserting if the output is correct
    assert isinstance(result, pd.Series)
    assert result["2023-09-01"] == 0.6

    # Asserting if the right methods were called
    mock_api.get_models.assert_called_once()
    mock_api.upload_diagnostics.assert_called_once_with(df=df, model_id='test_model_id')
    mock_api.diagnostics.assert_called()


def test_await_diagnostics_timeout(mock_api):
    obj = NumeraiSignalsEvaluator()
    mock_api.diagnostics.return_value = [{"status": "not_done"}]  # Simulate timeout scenario
    
    with pytest.raises(Exception, match=r"Diagnostics couldn't be retrieved within .* minutes after uploading."):
        obj._NumeraiSignalsEvaluator__await_diagnostics(api=mock_api, model_id="test_model_id", diagnostics_id="test_diag_id", timeout_min=0.001, interval_sec=2)


def test_get_raw_feature_exposures_pearson(classic_test_data):
    evaluator = NumeraiClassicEvaluator(era_col="era", fast_mode=False)
    np.random.seed(1)
    classic_test_data["prediction"] = np.random.uniform(size=len(classic_test_data))

    feature_list = [col for col in classic_test_data.columns if col.startswith("feature")]
    raw_exposures = evaluator.get_feature_exposures_pearson(classic_test_data, pred_col="prediction", feature_list=feature_list, cpu_cores=3)
    assert isinstance(raw_exposures, pd.DataFrame)
    assert len(raw_exposures) == len(classic_test_data["era"].unique())
    # Check that values are between -1 and 1
    assert raw_exposures.min().min() >= -1
    assert raw_exposures.max().max() <= 1
    for feature in feature_list:
        assert feature in raw_exposures.columns


def test_get_feature_exposures_corrv2(classic_test_data):
    evaluator = NumeraiClassicEvaluator(era_col="era", fast_mode=False)
    np.random.seed(1)
    classic_test_data["prediction"] = np.random.uniform(size=len(classic_test_data))

    feature_list = [col for col in classic_test_data.columns if col.startswith("feature")]
    raw_exposures = evaluator.get_feature_exposures_corrv2(classic_test_data, pred_col="prediction", feature_list=feature_list, cpu_cores=3)
    assert isinstance(raw_exposures, pd.DataFrame)
    assert len(raw_exposures) == len(classic_test_data["era"].unique())
    # Check that values are between -1 and 1
    assert raw_exposures.min().min() >= -1
    assert raw_exposures.max().max() <= 1
    for feature in feature_list:
        assert feature in raw_exposures.columns
    
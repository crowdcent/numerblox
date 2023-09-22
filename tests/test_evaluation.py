import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from numerblox.misc import Key
from numerblox.evaluation import NumeraiClassicEvaluator, NumeraiSignalsEvaluator

from utils import create_signals_sample_data

BASE_STATS_COLS = ['target', 'mean', 'std', 'sharpe', 'max_drawdown', 'apy', 'calmar_ratio', 'corr_with_example_preds', 'legacy_mean', 'legacy_std', 'legacy_sharpe', 'max_feature_exposure', 'feature_neutral_mean', 'feature_neutral_std', 'feature_neutral_sharpe', 'tb200_mean', 'tb200_std', 'tb200_sharpe', 'tb500_mean', 'tb500_std', 'tb500_sharpe', 'exposure_dissimilarity']
CLASSIC_SPECIFIC_STATS_COLS = ['feature_neutral_mean_v3', 'feature_neutral_std_v3', 'feature_neutral_sharpe_v3']

CLASSIC_STATS_COLS = BASE_STATS_COLS + CLASSIC_SPECIFIC_STATS_COLS
SIGNALS_STATS_COLS = BASE_STATS_COLS

df = create_signals_sample_data

def test_numerai_classic_evaluator():
    df = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random"] = np.random.uniform(size=len(df))

    evaluator = NumeraiClassicEvaluator(era_col="era", fast_mode=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
        example_col="prediction_random",
    )
    for col in CLASSIC_STATS_COLS:
        assert col in val_stats.columns
    for col in CLASSIC_SPECIFIC_STATS_COLS:
        val_stats[col]

def test_numerai_signals_evaluator(df):
    evaluator = NumeraiSignalsEvaluator(era_col="date", fast_mode=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
        example_col="prediction_random",
    )
    for col in SIGNALS_STATS_COLS:
        assert col in val_stats.columns

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

def test_get_neutralized_corr(df, mock_api):
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

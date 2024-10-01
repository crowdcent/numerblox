from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from numerblox.evaluation import ALL_CLASSIC_METRICS, FAST_METRICS, NumeraiClassicEvaluator, NumeraiSignalsEvaluator
from numerblox.misc import Key


@pytest.fixture()
def classic_test_data():
    return pd.read_parquet("tests/test_assets/val_3_eras.parquet")


@pytest.fixture
def create_signals_sample_data():
    instances = []
    tickers = ["ABC.US", "DEF.US", "GHI.US", "JKL.US", "MNO.US"]
    for ticker in tickers:
        price = np.random.randint(10, 100)
        for i in range(100):
            price += np.random.uniform(-1, 1)
            instances.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "open": price - 0.05,
                    "high": price + 0.02,
                    "low": price - 0.01,
                    "close": price,
                    "adjusted_close": price * np.random.uniform(0.5, 1.5),
                    "volume": np.random.randint(1000, 10000),
                    "target": np.random.uniform(),
                    "target_2": np.random.uniform(),
                    "prediction": np.random.uniform(),
                    "prediction_random": np.random.uniform(),
                }
            )
    # Add instances with only 10 days of data
    unwanted_tickers = ["XYZ.US", "RST.US", "UVW.US"]
    price = np.random.randint(10, 100)
    for ticker in unwanted_tickers:
        for i in range(10):
            price += np.random.uniform(-1, 1)
            instances.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "open": price - 0.05,
                    "high": price + 0.02,
                    "low": price - 0.01,
                    "close": price,
                    "adjusted_close": price * np.random.uniform(0.5, 1.5),
                    "volume": np.random.randint(1000, 10000),
                    "target": np.random.uniform(),
                    "target_2": np.random.uniform(),
                    "prediction": np.random.uniform(),
                    "prediction_random": np.random.uniform(),
                }
            )
    return pd.DataFrame(instances)


BASE_STATS_COLS = ["target", "mean", "std", "sharpe", "apy", "max_drawdown", "calmar_ratio"]
MAIN_CLASSIC_STATS_COLS = BASE_STATS_COLS + [
    "autocorrelation",
    "max_feature_exposure",
    "smart_sharpe",
    "legacy_mean",
    "legacy_std",
    "legacy_sharpe",
    "feature_neutral_mean_v3",
    "feature_neutral_std_v3",
    "feature_neutral_sharpe_v3",
    "feature_neutral_mean",
    "feature_neutral_std",
    "feature_neutral_sharpe",
    "feature_neutral_sharpe",
    "tb200_mean",
    "tb200_std",
    "tb200_sharpe",
    "tb500_mean",
    "tb500_std",
    "tb500_sharpe",
    "churn",
    "tb200_churn",
    "tb500_churn",
]


def test_numerai_classic_evaluator_fast_metrics(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random"] = np.random.uniform(size=len(df))

    evaluator = NumeraiClassicEvaluator(era_col="era", show_detailed_progress_bar=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
    )
    for col in BASE_STATS_COLS:
        assert col in val_stats.columns
        assert val_stats[col].iloc[0] != np.nan


def test_numerai_classic_evaluator_all_metrics(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random"] = np.random.uniform(size=len(df))
    df.loc[:, "prediction_random2"] = np.random.uniform(size=len(df))
    benchmark_cols = ["prediction_random", "prediction_random2"]
    evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=ALL_CLASSIC_METRICS, show_detailed_progress_bar=True)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction"],
        benchmark_cols=benchmark_cols,
    )
    # Add Benchmark specific stats
    BENCHMARK_STATS = []
    for col in benchmark_cols:
        BENCHMARK_STATS.extend(
            [
                f"mean_vs_{col}",
                f"std_vs_{col}",
                f"sharpe_vs_{col}",
                f"mc_mean_{col}",
                f"mc_std_{col}",
                f"mc_sharpe_{col}",
                f"corr_with_{col}",
                f"legacy_mc_mean_{col}",
                f"legacy_mc_std_{col}",
                f"legacy_mc_sharpe_{col}",
                f"exposure_dissimilarity_pearson_{col}",
                f"exposure_dissimilarity_spearman_{col}",
            ]
        )
    for col in MAIN_CLASSIC_STATS_COLS + BENCHMARK_STATS:
        assert col in val_stats.columns
        assert val_stats[col].iloc[0] != np.nan


def test_full_evaluation_out_of_bounds_pred(classic_test_data):
    df = classic_test_data
    # Invalid pred range
    df.loc[:, "prediction"] = np.random.uniform(size=len(df)) + 10
    evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=ALL_CLASSIC_METRICS, show_detailed_progress_bar=True)
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
        )
    # Invalid pred range
    df.loc[:, "prediction"] = np.random.uniform(size=len(df)) - 50
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
        )


def test_full_evaluation_out_of_bounds_target(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    # Invalid target range
    df["target"] = df["target"] + 10

    evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=ALL_CLASSIC_METRICS, show_detailed_progress_bar=True)
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
        )
    # Invalid target range
    df["target"] = df["target"] - 50
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
        )


def test_full_evaluation_out_of_bounds_benchmark(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))
    # Invalid benchmark range
    df.loc[:, "prediction_random2"] = np.random.uniform(size=len(df)) + 10

    evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=ALL_CLASSIC_METRICS, show_detailed_progress_bar=True)
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
            benchmark_cols=["prediction_random2"],
        )
    # Invalid benchmark range
    df.loc[:, "prediction_random2"] = np.random.uniform(size=len(df)) - 50
    with pytest.raises(AssertionError):
        evaluator.full_evaluation(
            dataf=df,
            target_col="target",
            pred_cols=["prediction"],
            benchmark_cols=["prediction_random2"],
        )


def test_numerai_signals_evaluator(create_signals_sample_data):
    df = create_signals_sample_data
    evaluator = NumeraiSignalsEvaluator(era_col="date", metrics_list=FAST_METRICS, show_detailed_progress_bar=False)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction", "prediction_random"],
    )
    for col in BASE_STATS_COLS:
        assert col in val_stats.columns
        assert val_stats[col].iloc[0] != np.nan


def test_classic_evaluator_wrong_metrics_list():
    with pytest.raises(AssertionError):
        _ = NumeraiClassicEvaluator(era_col="era", metrics_list=["mean_std_sharpe", "invalid_metric"])


def test_signals_evaluator_wrong_metrics_list():
    with pytest.raises(AssertionError):
        _ = NumeraiSignalsEvaluator(era_col="era", metrics_list=["mean_std_sharpe", "invalid_metric"])


def test_evaluator_custom_functions(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))

    def custom_func(dataf, target_col, pred_col, custom_arg: int):
        """Simple example func: Mean of residuals."""
        return np.mean(dataf[target_col] - dataf[pred_col])

    def average_col_stats_func(**kwargs):
        """Averaging stats."""
        return sum(kwargs.values()) / len(kwargs)

    def mean_mean(col_stats):
        """Average means."""
        return (col_stats["mean"] + col_stats["tb200_mean"] + col_stats["tb500_mean"]) / 3

    custom_functions = {
        "residuals": {
            "func": custom_func,
            "args": {
                "dataf": "dataf",  # String referring to a local variable
                "pred_col": "pred_col",
                "target_col": "target_col",
                "custom_arg": 15,  # Literal arg
            },
            # List of local variables to use/resolve
            "local_args": ["dataf", "pred_col", "target_col"],
        },
        "tb500_sharpe": {"func": average_col_stats_func, "args": {"tb500_sharpe": "tb500_sharpe", "tb200_sharpe": "tb200_sharpe"}, "local_args": ["tb500_sharpe", "tb200_sharpe"]},
        "tb500_mean": {"func": average_col_stats_func, "args": {"tb500_mean": "tb500_mean", "tb200_mean": "tb200_mean"}, "local_args": ["tb500_mean", "tb200_mean"]},
        "mean_of_means": {
            "func": mean_mean,
            "args": {
                "col_stats": "col_stats",
            },
            "local_args": ["col_stats"],
        },
    }
    evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=["mean_std_sharpe", "tb500_mean_std_sharpe", "tb200_mean_std_sharpe"], custom_functions=custom_functions, show_detailed_progress_bar=True)
    val_stats = evaluator.full_evaluation(
        dataf=df,
        target_col="target",
        pred_cols=["prediction"],
    )
    for custom_cols in custom_functions.keys():
        assert custom_cols in val_stats.columns
        assert val_stats[custom_cols].iloc[0] != np.nan


def test_evaluator_invalid_custom_function(classic_test_data):
    df = classic_test_data
    df.loc[:, "prediction"] = np.random.uniform(size=len(df))

    # Invalid function dict (args missing)
    custom_functions = {
        "custom_func": {
            "func": "",
        }
    }

    # Initialization fails if input is invalid (For example missing args)
    with pytest.raises(ValueError):
        NumeraiClassicEvaluator(era_col="era", custom_functions=custom_functions)


@pytest.fixture
def mock_api():
    with patch("numerblox.evaluation.SignalsAPI") as mock:
        mock_instance = mock.return_value

        # Mock get_models method
        mock_instance.get_models.return_value = {"test_model": "test_model_id"}

        # Mock upload_diagnostics method
        mock_instance.upload_diagnostics.return_value = "test_diag_id"

        # Mock diagnostics method
        mock_instance.diagnostics.return_value = [{"status": "done", "perEraDiagnostics": [{"era": "2023-09-01", "validationFncV4": 0.6, "validationCorrV4": 0.6, "validationIcV2": 0.3, "validationRic": 0.5}]}]
        yield mock_instance


def test_get_diagnostics(create_signals_sample_data, mock_api):
    df = create_signals_sample_data
    obj = NumeraiSignalsEvaluator(era_col="date")
    result = obj.get_diagnostics(df, "test_model", Key("Hello", "World"), col="validationIcV2")

    # Asserting if the output is correct
    assert isinstance(result, pd.DataFrame)
    assert result.loc["2023-09-01"]["validationIcV2"] == 0.3

    # Asserting if the right methods were called
    mock_api.get_models.assert_called_once()
    mock_api.upload_diagnostics.assert_called_once_with(df=df, model_id="test_model_id")
    mock_api.diagnostics.assert_called()


def test_get_diagnostics_all_cols(create_signals_sample_data, mock_api):
    df = create_signals_sample_data
    obj = NumeraiSignalsEvaluator(era_col="date")
    result = obj.get_diagnostics(df, "test_model", Key("Hello", "World"), col=None)

    # Asserting if the output is correct
    assert isinstance(result, pd.DataFrame)
    for col in obj.VALID_DIAGNOSTICS_COLS:
        assert col in result.columns
    assert result.loc["2023-09-01"]["validationCorrV4"] == 0.6
    assert result.loc["2023-09-01"]["validationIcV2"] == 0.3
    assert result.loc["2023-09-01"]["validationRic"] == 0.5
    assert result.loc["2023-09-01"]["validationFncV4"] == 0.6


def test_get_diagnostics_invalid_col(create_signals_sample_data):
    df = create_signals_sample_data
    obj = NumeraiSignalsEvaluator(era_col="date")
    with pytest.raises(AssertionError):
        obj.get_diagnostics(df, "test_model", Key("Hello", "World"), col="invalid_corr_col")


def test_await_diagnostics_timeout(mock_api):
    obj = NumeraiSignalsEvaluator()
    mock_api.diagnostics.return_value = [{"status": "not_done"}]  # Simulate timeout scenario

    with pytest.raises(Exception, match=r"Diagnostics couldn't be retrieved within .* minutes after uploading."):
        obj._NumeraiSignalsEvaluator__await_diagnostics(api=mock_api, model_id="test_model_id", diagnostics_id="test_diag_id", timeout_min=0.001, interval_sec=2)


def test_get_raw_feature_exposures_pearson(classic_test_data):
    evaluator = NumeraiClassicEvaluator(era_col="era")
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
    evaluator = NumeraiClassicEvaluator(era_col="era")
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

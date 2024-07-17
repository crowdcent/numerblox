import os
import pytest
import pandas as pd
from uuid import uuid4
from pathlib import PosixPath

from numerapi import NumerAPI, SignalsAPI
from numerblox.download import NumeraiClassicDownloader, NumeraiSignalsDownloader, KaggleDownloader, EODDownloader

ALL_CLASSIC_VERSIONS = set(s.split("/")[0] for s in NumerAPI().list_datasets() if not "signals" in s)
ALL_SIGNALS_VERSIONS = set(s.replace("signals/", "").split("/")[0] for s in SignalsAPI().list_datasets() if s.startswith("signals/v"))
TEST_CLASSIC_DIR = f"test_numclassic_general_{uuid4()}"
TEST_SIGNALS_DIR = f"test_numsignals_general_{uuid4()}"
# Train and val data available for v5.0
TEST_CLASSIC_VERSIONS = ["4.2", "4.3", "5.0"]
# Live data for v5.0 will be released in September 2024
TEST_CLASSIC_LIVE_VERSIONS = ["4.2", "4.3"]
TEST_SIGNALS_VERSIONS = ["1.0"]

def test_base():
    numer_classic_downloader = NumeraiClassicDownloader(TEST_CLASSIC_DIR)

    # Test building class
    assert isinstance(numer_classic_downloader.dir, PosixPath)
    assert numer_classic_downloader.dir.is_dir()

    # Test is_empty
    (numer_classic_downloader.dir / "test.txt").write_text("test")
    assert not numer_classic_downloader.is_empty

    # Remove contents
    numer_classic_downloader.remove_base_directory()
    assert not os.path.exists(TEST_CLASSIC_DIR)


def test_classic():
    dl = NumeraiClassicDownloader(TEST_CLASSIC_DIR)

    # Check versions
    assert dl.dataset_versions == ALL_CLASSIC_VERSIONS

    # Test live download
    for version in TEST_CLASSIC_LIVE_VERSIONS:
        dl.download_live_data("live", version=version)
        assert os.path.exists(dl.dir / "live")
        assert os.path.exists(dl.dir / "live" / "live_int8.parquet")

        # Test example data
        dl.download_example_data("test/", version=version)
        assert os.path.exists(dl.dir / "test")
        assert os.path.exists(dl.dir / "test" / "live_example_preds.parquet")
        assert os.path.exists(dl.dir / "test" / "validation_example_preds.parquet")

    # Test features
    features = dl.get_classic_features()
    assert isinstance(features, dict)
    assert len(features["feature_sets"]["medium"]) == 705
    # Check that feature_stats and feature_sets keys exist
    assert "feature_stats" in features.keys()
    assert "feature_sets" in features.keys()

    # Test metamodel preds
    meta_model = dl.download_meta_model_preds()
    assert os.path.exists(dl.dir / "meta_model.parquet")
    assert isinstance(meta_model, pd.DataFrame)
    assert "numerai_meta_model" in meta_model.columns

    dl.remove_base_directory()


def test_signals():
    dl = NumeraiSignalsDownloader(TEST_SIGNALS_DIR)

    # Check versions
    assert dl.dataset_versions == ALL_SIGNALS_VERSIONS

    # Test live download
    for version in TEST_SIGNALS_VERSIONS:
        dl.download_live_data("live", version=version)
        assert os.path.exists(dl.dir / "live")
        assert os.path.exists(dl.dir / "live" / "live.parquet")

        # Test example data
        dl.download_example_data("test/", version=version)
        assert os.path.exists(dl.dir / "test")
        assert os.path.exists(dl.dir / "test" / "live_example_preds.parquet")
        assert os.path.exists(dl.dir / "test" / "validation_example_preds.parquet")

    dl.remove_base_directory()


def test_classic_versions():
    downloader = NumeraiClassicDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["3", "6.8"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_live_data(version=version)
            
    downloader.remove_base_directory()


def test_signals_versions():
    downloader = NumeraiSignalsDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["0", "0.5", "3.5"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_live_data(version=version)
            
    downloader.remove_base_directory()


def test_kaggle_downloader():
    try:
        kd = KaggleDownloader(f"test_kaggle_{uuid4()}")
        assert os.path.exists(kd.dir)
        kd.remove_base_directory()
    except OSError:
        pass


def test_eod():
    eod = EODDownloader(f"test_eod_{uuid4()}", key="DEMO", tickers=["AAPL.US"])
    eod.download_live_data()
    eod.download_training_data()
    eod.remove_base_directory()
    
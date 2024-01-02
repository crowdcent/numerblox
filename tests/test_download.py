import os
import pytest
import pandas as pd
from uuid import uuid4
from pathlib import PosixPath

from numerapi import NumerAPI
from numerblox.download import NumeraiClassicDownloader, KaggleDownloader, EODDownloader

ALL_DATASET_VERSIONS = set(s.split("/")[0] for s in NumerAPI().list_datasets())
ALL_DATASET_VERSIONS.discard("signals")
TEST_DIR = f"test_numclassic_general_{uuid4()}"
TEST_VERSIONS = ["4.2", "4.3"]

def test_base():
    numer_classic_downloader = NumeraiClassicDownloader(TEST_DIR)

    # Test building class
    assert isinstance(numer_classic_downloader.dir, PosixPath)
    assert numer_classic_downloader.dir.is_dir()

    # Test is_empty
    (numer_classic_downloader.dir / "test.txt").write_text("test")
    assert not numer_classic_downloader.is_empty

    # Remove contents
    numer_classic_downloader.remove_base_directory()
    assert not os.path.exists(TEST_DIR)

def test_classic():
    dl = NumeraiClassicDownloader(TEST_DIR)

    # Check versions
    assert dl.dataset_versions == ALL_DATASET_VERSIONS

    # Test inference download
    for version in TEST_VERSIONS:
        dl.download_inference_data("inference", version=version)
        assert os.path.exists(dl.dir / "inference")
        assert os.path.exists(dl.dir / "inference" / "live_int8.parquet")

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

def test_classic_versions():
    downloader = NumeraiClassicDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["3", "5", "6.8"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_inference_data(version=version)
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
    eod.download_inference_data()
    eod.download_training_data()
    eod.remove_base_directory()
    
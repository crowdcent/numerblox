import os
import pytest
from uuid import uuid4
from pathlib import PosixPath

from numerblox.download import NumeraiClassicDownloader, KaggleDownloader, EODDownloader

CURRENT_VERSION = "4.2"
ALL_DATASET_VERSIONS = ["4", "4.1", "4.2"]
TEST_DIR = f"test_numclassic_general_{uuid4()}"

def test_base():
    numer_classic_downloader = NumeraiClassicDownloader(TEST_DIR)

    # Test building class
    assert isinstance(numer_classic_downloader.dir, PosixPath)
    assert numer_classic_downloader.dir.is_dir()

    # Test is_empty
    (numer_classic_downloader.dir / "test.txt").write_text("test")
    print(f"Directory contents:\n{numer_classic_downloader.get_all_files}")
    assert not numer_classic_downloader.is_empty

    # Remove contents
    numer_classic_downloader.remove_base_directory()
    assert not os.path.exists(TEST_DIR)

def test_classic():
    dl = NumeraiClassicDownloader(TEST_DIR)

    # Check versions
    assert list(dl.version_mapping.keys()) == ALL_DATASET_VERSIONS

    # Test inference download
    dl.download_inference_data("inference", version=CURRENT_VERSION, int8=True)

    # Check that you can't use int8=False with v4.2.
    funcs = [
    (dl.download_inference_data, "inference"),
    (dl.download_training_data, "training")
    ]
    for func, arg in funcs:
        try:
            func(arg, version="4.2", int8=False)
        except NotImplementedError:
            pass

    # Test example data
    dl.download_example_data("test/", version=CURRENT_VERSION)
    assert os.path.exists(dl.dir / "test")
    assert os.path.exists(dl.dir / "test" / "live_example_preds.parquet")
    assert os.path.exists(dl.dir / "test" / "validation_example_preds.parquet")

    # Test features
    features = dl.get_classic_features()
    assert isinstance(features, dict)
    assert len(features["feature_sets"]["medium"]) == 583
    # Check that feature_stats and feature_sets keys exist
    assert "feature_stats" in features.keys()
    assert "feature_sets" in features.keys()

    # Test metamodel preds
    dl.download_meta_model_preds()
    assert os.path.exists(dl.dir / "meta_model.parquet")

    dl.remove_base_directory()

def test_classic_version_mapping():
    downloader = NumeraiClassicDownloader(directory_path="some_path")

    # Test supported versions
    supported_versions = ["4", "4.1", "4.2"]
    for version in supported_versions:
        mapping = downloader._get_version_mapping(version)
        assert isinstance(mapping, dict), f"Mapping for version {version} should return a dictionary."
        assert "train" in mapping, f"Version {version} mapping should contain 'train' key."
        assert "inference" in mapping, f"Version {version} mapping should contain 'inference' key."
        assert "live" in mapping, f"Version {version} mapping should contain 'live' key."
        assert "example" in mapping, f"Version {version} mapping should contain 'example' key."

    # Test unsupported versions
    unsupported_versions = ["3", "5", "4.3"]
    for version in unsupported_versions:
        with pytest.raises(NotImplementedError, match=f"Version '{version}' is not available"):
            downloader._get_version_mapping(version)

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
    
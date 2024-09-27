import os
import pytest
from uuid import uuid4
from pathlib import PosixPath
from numerapi import NumerAPI

from numerblox.download import NumeraiClassicDownloader


ALL_CLASSIC_VERSIONS = set(s.split("/")[0] for s in NumerAPI().list_datasets() if not "signals" in s)

TEST_CLASSIC_DIR = f"test_numclassic_general_{uuid4()}"
TEST_CLASSIC_VERSIONS = ["5.0"]

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
    for version in TEST_CLASSIC_VERSIONS:
        dl.download_live_data("live", version=version)
        assert os.path.exists(dl.dir / "live")
        assert os.path.exists(dl.dir / "live" / "live.parquet")

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
    assert "feature_sets" in features.keys()

    dl.remove_base_directory()

def test_classic_versions():
    downloader = NumeraiClassicDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["3"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_live_data(version=version)
            
    downloader.remove_base_directory()

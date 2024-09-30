import os
from uuid import uuid4

import pytest
from numerapi import SignalsAPI

from numerblox.download import EODDownloader, KaggleDownloader, NumeraiSignalsDownloader

ALL_SIGNALS_VERSIONS = set(s.replace("signals/", "").split("/")[0] for s in SignalsAPI().list_datasets() if s.startswith("signals/v"))
TEST_SIGNALS_DIR = f"test_numsignals_general_{uuid4()}"
TEST_SIGNALS_VERSIONS = ["2.0"]


@pytest.mark.xfail(reason="May fail due to API rate limiting")
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


@pytest.mark.xfail(reason="May fail due to API rate limiting")
def test_signals_versions():
    downloader = NumeraiSignalsDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["0"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_live_data(version=version)

    downloader.remove_base_directory()


@pytest.mark.xfail(reason="May fail due to API rate limiting or missing credentials")
def test_kaggle_downloader():
    try:
        kd = KaggleDownloader(f"test_kaggle_{uuid4()}")
        assert os.path.exists(kd.dir)
        kd.remove_base_directory()
    except OSError:
        pass


@pytest.mark.xfail(reason="May fail due to API rate limiting or missing credentials")
def test_eod():
    eod = EODDownloader(f"test_eod_{uuid4()}", key="DEMO", tickers=["AAPL.US"])
    eod.download_live_data()
    eod.download_training_data()
    eod.remove_base_directory()

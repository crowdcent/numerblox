import os
import pytest
from uuid import uuid4

from numerblox.download import NumeraiCryptoDownloader


ALL_CRYPTO_VERSIONS = ["v1.0"]

@pytest.mark.xfail(reason="May fail due to API rate limiting")
def test_crypto():
    TEST_CRYPTO_DIR = f"test_numcrypto_general_{uuid4()}"
    dl = NumeraiCryptoDownloader(TEST_CRYPTO_DIR)

    # Check versions
    assert dl.dataset_versions == ALL_CRYPTO_VERSIONS

    # Test live download
    dl.download_live_data("live", version="1.0")
    assert os.path.exists(dl.dir / "live")
    assert os.path.exists(dl.dir / "live" / "live_universe.parquet")

    # Test training data download
    dl.download_training_data("train/", version="1.0")
    assert os.path.exists(dl.dir / "train")
    assert os.path.exists(dl.dir / "train" / "train_targets.parquet")

@pytest.mark.xfail(reason="May fail due to API rate limiting")
def test_crypto_versions():
    downloader = NumeraiCryptoDownloader(directory_path=f"some_path_{uuid4()}")

    # Test unsupported versions
    unsupported_versions = ["0", "0.5", "3.5"]
    for version in unsupported_versions:
        with pytest.raises(AssertionError):
            downloader.download_training_data(version=version)
        with pytest.raises(AssertionError):
            downloader.download_live_data(version=version)

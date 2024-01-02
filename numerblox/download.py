import os
import time
import glob
import json
import shutil
import concurrent
import pandas as pd
from tqdm.auto import tqdm
from numerapi import NumerAPI
from google.cloud import storage
from datetime import datetime as dt
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dateutil.relativedelta import relativedelta

from .numerframe import NumerFrame


class BaseIO(ABC):
    """
    Basic functionality for IO (downloading and uploading).

    :param directory_path: Base folder for IO. Will be created if it does not exist.
    """
    def __init__(self, directory_path: str):
        self.dir = Path(directory_path)
        self._create_directory()

    def remove_base_directory(self):
        """Remove directory with all contents."""
        abs_path = self.dir.resolve()
        print(
            f"WARNING: Deleting directory for '{self.__class__.__name__}'\nPath: '{abs_path}'"
        )
        shutil.rmtree(abs_path)

    def download_file_from_gcs(self, bucket_name: str, gcs_path: str):
        """
        Get file from GCS bucket and download to local directory.
        :param gcs_path: Path to file on GCS bucket.
        """
        blob_path = str(self.dir.resolve())
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=blob_path)
        blob.download_to_filename(gcs_path)
        print(
            f"Downloaded GCS object '{gcs_path}' from bucket '{blob.bucket.id}' to local directory '{blob_path}'."
        )

    def upload_file_to_gcs(self, bucket_name: str, gcs_path: str, local_path: str):
        """
        Upload file to some GCS bucket.
        :param gcs_path: Path to file on GCS bucket.
        """
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=gcs_path)
        blob.upload_from_filename(local_path)
        print(
            f"Local file '{local_path}' uploaded to '{gcs_path}' in bucket {blob.bucket.id}"
        )

    def download_directory_from_gcs(self, bucket_name: str, gcs_path: str):
        """
        Copy full directory from GCS bucket to local environment.
        :param gcs_path: Name of directory on GCS bucket.
        """
        blob_path = str(self.dir.resolve())
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=blob_path)
        for gcs_file in glob.glob(gcs_path + "/**", recursive=True):
            if os.path.isfile(gcs_file):
                blob.download_to_filename(blob_path)
        print(
            f"Directory '{gcs_path}' from bucket '{blob.bucket.id}' downloaded to '{blob_path}'"
        )

    def upload_directory_to_gcs(self, bucket_name: str, gcs_path: str):
        """
        Upload full base directory to GCS bucket.
        :param gcs_path: Name of directory on GCS bucket.
        """
        blob = self._get_gcs_blob(bucket_name=bucket_name, blob_path=gcs_path)
        for local_path in glob.glob(str(self.dir) + "/**", recursive=True):
            if os.path.isfile(local_path):
                blob.upload_from_filename(local_path)
        print(
            f"Directory '{self.dir}' uploaded to '{gcs_path}' in bucket {blob.bucket.id}"
        )

    def _get_gcs_blob(self, bucket_name: str, blob_path: str) -> storage.Blob:
        """ Create blob that interacts with Google Cloud Storage (GCS). """
        client = storage.Client()
        # https://console.cloud.google.com/storage/browser/[bucket_name]
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob

    def _append_folder(self, folder: str) -> Path:
        """
        Return base directory Path object appended with 'folder'.
        Create directory if it does not exist.
        """
        dir = Path(self.dir / folder)
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def _create_directory(self):
        """ Create base directory if it does not exist. """
        if not self.dir.is_dir():
            print(
                f"No existing directory found at '{self.dir}'. Creating directory..."
            )
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def get_all_files(self) -> list:
        """ Return all paths of contents in directory. """
        return list(self.dir.iterdir())

    @property
    def is_empty(self) -> bool:
        """ Check if directory is empty. """
        return not bool(self.get_all_files)


class BaseDownloader(BaseIO):
    """
    Abstract base class for downloaders.

    :param directory_path: Base folder to download files to.
    """
    def __init__(self, directory_path: str):
        super().__init__(directory_path=directory_path)

    @abstractmethod
    def download_training_data(self, *args, **kwargs):
        """ Download all necessary files needed for training. """
        ...

    @abstractmethod
    def download_inference_data(self, *args, **kwargs):
        """ Download minimal amount of files needed for weekly inference. """
        ...

    @staticmethod
    def _load_json(file_path: str, verbose=False, *args, **kwargs) -> dict:
        """ Load JSON from file and return as dictionary. """
        with open(Path(file_path)) as json_file:
            json_data = json.load(json_file, *args, **kwargs)
        if verbose:
            print(json_data)
        return json_data

    def _default_save_path(self, start: dt, end: dt, backend: str):
        """ Save to downloader directory indicating backend, start date and end date as parquet file. """
        return f"{self.dir}/{backend}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"

    def __call__(self, *args, **kwargs):
        """
        The most common use case will be to get weekly inference data. So calling the class itself returns inference data.
        """
        self.download_inference_data(*args, **kwargs)


class NumeraiClassicDownloader(BaseDownloader):
    """
    WARNING: Versions 1-3 (legacy data) are deprecated. Only supporting version 4+.

    Downloading from NumerAPI for Numerai Classic data. \n
    :param directory_path: Base folder to download files to. \n
    All *args, **kwargs will be passed to NumerAPI initialization.
    """
    TRAIN_DATASET_NAME = "train_int8.parquet"
    VALIDATION_DATASET_NAME = "validation_int8.parquet"
    LIVE_DATASET_NAME = "live_int8.parquet"
    LIVE_EXAMPLE_PREDS_NAME = "live_example_preds.parquet"
    VALIDATION_EXAMPLE_PREDS_NAME = "validation_example_preds.parquet"

    def __init__(self, directory_path: str, *args, **kwargs):
        super().__init__(directory_path=directory_path)
        self.napi = NumerAPI(*args, **kwargs)
        self.current_round = self.napi.get_current_round()
        # Get all available versions available for Numerai.
        self.dataset_versions = set(s.split("/")[0] for s in NumerAPI().list_datasets())
        self.dataset_versions.discard("signals")

    def download_training_data(
        self, subfolder: str = "", version: str = "4.3"
    ):
        """
        Get Numerai classic training and validation data.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        4 = April 2022 dataset
        4.1 = Sunshine dataset
        4.2 (default) = Rain Dataset
        4.3 = Midnight dataset
        """
        self._check_dataset_version(version)
        train_val_files = [f"v{version}/{self.TRAIN_DATASET_NAME}",
                           f"v{version}/{self.VALIDATION_DATASET_NAME}"]
        for file in train_val_files:
            dest_path = self.__get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path
            )

    def download_inference_data(
        self,
        subfolder: str = "",
        version: str = "4.3",
        round_num: int = None,
    ):
        """
        Get Numerai classic inference (tournament) data.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        4 = April 2022 dataset
        4.1 = Sunshine dataset
        4.2 (default) = Rain Dataset
        4.3 = Midnight dataset
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        self.download_live_data(subfolder=subfolder, version=version, round_num=round_num)

    def download_single_dataset(
        self, filename: str, dest_path: str, round_num: int = None
    ):
        """
        Download one of the available datasets through NumerAPI.

        :param filename: Name as listed in NumerAPI (Check NumerAPI().list_datasets() for full overview)
        :param dest_path: Full path where file will be saved.
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        print(
            f"Downloading '{filename}'."
        )
        self.napi.download_dataset(
            filename=filename,
            dest_path=dest_path,
            round_num=round_num
        )

    def download_live_data(
            self,
            subfolder: str = "",
            version: str = "4.3",
            round_num: int = None
    ):
        """
        Download all live data in specified folder for given version (i.e. minimal data needed for inference).

        :param subfolder: Specify folder to create folder within directory root.
        Saves in directory root by default.
        :param version: Numerai dataset version. 
        4 = April 2022 dataset
        4.1 = Sunshine dataset
        4.2 (default) = Rain Dataset
        4.3 = Midnight dataset
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        self._check_dataset_version(version)
        live_files = [f"v{version}/{self.LIVE_DATASET_NAME}"]
        for file in live_files:
            dest_path = self.__get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
                round_num=round_num
            )

    def download_example_data(
        self, subfolder: str = "", version: str = "4.3", round_num: int = None
    ):
        """
        Download all example prediction data in specified folder for given version.

        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param version: Numerai dataset version.
        4 = April 2022 dataset
        4.1 = Sunshine dataset
        4.2 (default) = Rain Dataset
        4.3 = Midnight dataset
        :param round_num: Numerai tournament round number. Downloads latest round by default.
        """
        self._check_dataset_version(version)
        example_files = [f"v{version}/{self.LIVE_EXAMPLE_PREDS_NAME}", 
                         f"v{version}/{self.VALIDATION_EXAMPLE_PREDS_NAME}"]
        for file in example_files:
            dest_path = self.__get_dest_path(subfolder, file)
            self.download_single_dataset(
                filename=file,
                dest_path=dest_path,
                round_num=round_num
            )

    def get_classic_features(self, subfolder: str = "", filename="v4.3/features.json", *args, **kwargs) -> dict:
        """
        Download feature overview (stats and feature sets) through NumerAPI and load as dict.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param filename: name for feature overview.
        *args, **kwargs will be passed to the JSON loader.
        :return: Feature overview dict
        """
        version = filename.split("/")[0].replace("v", "")
        self._check_dataset_version(version)
        dest_path = self.__get_dest_path(subfolder, filename)
        self.download_single_dataset(filename=filename,
                                     dest_path=dest_path)
        json_data = self._load_json(dest_path, *args, **kwargs)
        return json_data

    def download_meta_model_preds(self, subfolder: str = "", filename="v4.3/meta_model.parquet") -> pd.DataFrame:
        """
        Download Meta model predictions through NumerAPI.
        :param subfolder: Specify folder to create folder within base directory root.
        Saves in base directory root by default.
        :param filename: name for meta model predictions file.
        :return: Meta model predictions as DataFrame.
        """
        version = filename.split("/")[0].replace("v", "")
        self._check_dataset_version(version)
        dest_path = self.__get_dest_path(subfolder, filename)
        self.download_single_dataset(
            filename=filename,
            dest_path=dest_path,
            )
        return pd.read_parquet(dest_path)

    def __get_dest_path(self, subfolder: str, filename: str) -> str:
        """ Prepare destination path for downloading. """
        dir = self._append_folder(subfolder)
        dest_path = str(dir.joinpath(filename.split("/")[-1]))
        return dest_path
    
    def _check_dataset_version(self, version: str):
        assert f"v{version}" in self.dataset_versions, f"Version '{version}' is not available in NumerAPI."


class KaggleDownloader(BaseDownloader):
    """
    Download financial data from Kaggle.

    For authentication, make sure you have a directory called .kaggle in your home directory
    with therein a kaggle.json file. kaggle.json should have the following structure: \n
    `{"username": USERNAME, "key": KAGGLE_API_KEY}` \n
    More info on authentication: github.com/Kaggle/kaggle-api#api-credentials \n

    More info on the Kaggle Python API: kaggle.com/donkeys/kaggle-python-api \n

    :param directory_path: Base folder to download files to.
    """
    def __init__(self, directory_path: str):
        self.__check_kaggle_import()
        super().__init__(directory_path=directory_path)

    def download_inference_data(self, kaggle_dataset_path: str):
        """
        Download arbitrary Kaggle dataset.
        :param kaggle_dataset_path: Path on Kaggle (URL slug on kaggle.com/)
        """
        self.download_training_data(kaggle_dataset_path)

    def download_training_data(self, kaggle_dataset_path: str):
        """
        Download arbitrary Kaggle dataset.
        :param kaggle_dataset_path: Path on Kaggle (URL slug on kaggle.com/)
        """
        import kaggle
        kaggle.api.dataset_download_files(kaggle_dataset_path,
                                          path=self.dir, unzip=True)

    @staticmethod
    def __check_kaggle_import():
        try:
            import kaggle
        except OSError:
            raise OSError("Could not find kaggle.json credentials. Make sure it's located in /home/runner/.kaggle. Or use the environment method. Check github.com/Kaggle/kaggle-api#api-credentials for more information on authentication.")


class EODDownloader(BaseDownloader):
    """
    Download data from EOD historical data. \n
    More info: https://eodhistoricaldata.com/

    Make sure you have the underlying Python package installed.
    `pip install eod`.

    :param directory_path: Base folder to download files to. \n
    :param key: Valid EOD client key. \n
    :param tickers: List of valid EOD tickers (Bloomberg ticker format). \n
    :param frequency: Choose from [d, w, m]. \n
    Daily data by default.
    """
    def __init__(self,
                 directory_path: str,
                 key: str,
                 tickers: list,
                 frequency: str = "d"):
        super().__init__(directory_path=directory_path)
        self.key = key
        self.tickers = tickers
        try: 
            from eod import EodHistoricalData
        except ImportError:
            raise ImportError("Could not import eod package. Please install eod package with 'pip install eod'")
        self.client = EodHistoricalData(self.key)
        self.frequency = frequency
        self.current_time = dt.now()
        self.end_date = self.current_time.strftime("%Y-%m-%d")
        self.cpu_count = os.cpu_count()
        # Time to sleep in between API calls to avoid hitting EOD rate limits.
        # EOD rate limit is set at 1000 calls per minute.
        self.sleep_time = self.cpu_count / 32

    def download_inference_data(self):
        """ Download one year of data for defined tickers. """
        start = (pd.Timestamp(self.current_time) - relativedelta(years=1)).strftime("%Y-%m-%d")
        dataf = self.get_live_data(start=start)
        dataf.to_parquet(self._default_save_path(start=pd.Timestamp(start),
                                                 end=pd.Timestamp(self.end_date),
                                                 backend="eod"))

    def download_training_data(self, start: str = None):
        """
        Download full date length available.
        start: Starting data in %Y-%m-%d format.
        """
        start = start if start else "1970-01-01"
        dataf = self.generate_full_dataf(start=start)
        dataf.to_parquet(self._default_save_path(start=pd.Timestamp(start),
                                                 end=pd.Timestamp(self.end_date),
                                                 backend="eod"))

    def get_live_data(self, start: str) -> NumerFrame:
        """
        Get NumerFrame data from some starting date.
        start: Starting data in %Y-%m-%d format.
        """
        dataf = self.generate_full_dataf(start=start)
        return NumerFrame(dataf)

    def generate_full_dataf(self, start: str) -> pd.DataFrame:
        """
        Collect all price data for list of EOD ticker symbols (Bloomberg tickers).
        start: Starting data in %Y-%m-%d format.
        """
        price_datafs = []
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            tasks = [executor.submit(self.generate_stock_dataf, ticker, start) for ticker in self.tickers]
            for task in tqdm(concurrent.futures.as_completed(tasks),
                             total=len(self.tickers),
                             desc="EOD price data extraction"):
                price_datafs.append(task.result())
        return pd.concat(price_datafs)

    def generate_stock_dataf(self, ticker: str, start: str) -> pd.DataFrame:
        """
        Generate Price DataFrame for a single ticker.
        ticker: EOD ticker symbol (Bloomberg tickers).
        For example, Apple stock = AAPL.US.
        start: Starting data in %Y-%m-%d format.
        """
        time.sleep(self.sleep_time)
        try:
            resp = self.client.get_prices_eod(ticker, period=self.frequency,
                                              from_=start, to=self.end_date)
            stock_df = pd.DataFrame(resp).set_index('date')
            stock_df['ticker'] = ticker
        except Exception as e:
            print(f"WARNING: Date pull failed on ticker: '{ticker}'. Exception: {e}")
            stock_df = pd.DataFrame()
        return stock_df

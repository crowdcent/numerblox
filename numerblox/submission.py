import os
import uuid
import numpy as np
import pandas as pd
from typing import Union
from copy import deepcopy
from random import choices
from tqdm.auto import tqdm
from datetime import datetime
from abc import abstractmethod
from string import ascii_uppercase
from rich import print as rich_print
from numerapi import NumerAPI, SignalsAPI
from dateutil.relativedelta import relativedelta, FR

from .download import BaseIO
from .key import Key


class BaseSubmitter(BaseIO):
    """
    Basic functionality for submitting to Numerai. \n
    Uses numerapi under the hood.
    More info: https://numerapi.readthedocs.io/ \n

    :param directory_path: Directory to store and read submissions from. \n
    :param api: NumerAPI or SignalsAPI
    """
    def __init__(self, directory_path: str, api: Union[NumerAPI, SignalsAPI]):
        super().__init__(directory_path)
        self.api = api

    @abstractmethod
    def save_csv(
        self,
        dataf: pd.DataFrame,
        file_name: str,
        cols: Union[str, list],
        *args,
        **kwargs,
    ):
        """
        For Numerai Classic: Save index column + 'cols' (targets) to CSV.
        For Numerai Signals: Save ticker, friday_date, data_type and signal columns to CSV.
        """
        ...

    def upload_predictions(self, file_name: str, model_name: str, *args, **kwargs):
        """
        Upload CSV file to Numerai for given model name.
        :param file_name: File name/path relative to directory_path.
        :param model_name: Lowercase raw model name (For example, 'integration_test').
        """
        full_path = str(self.dir / file_name)
        model_id = self._get_model_id(model_name=model_name)
        api_type = str(self.api.__class__.__name__)
        rich_print(
            f":airplane: {api_type}: Uploading predictions from '{full_path}' for model [bold blue]'{model_name}'[/bold blue] (model_id='{model_id}') :airplane:"
        )
        self.api.upload_predictions(
            file_path=full_path, model_id=model_id, *args, **kwargs
        )
        rich_print(
            f":thumbs_up: {api_type} submission of '{full_path}' for [bold blue]{model_name}[/bold blue] is successful! :thumbs_up:"
        )

    def full_submission(
        self,
        dataf: pd.DataFrame,
        model_name: str,
        cols: Union[str, list],
        file_name: str = 'submission.csv',
        *args,
        **kwargs,
    ):
        """
        Save DataFrame to csv and upload predictions through API.

        :param dataf: Main DataFrame containing `cols`.
        :param model_name: Lowercase Numerai model name.
        :param file_name: path to save model to relative to base directory.
        :param cols: Columns to be saved in submission file.
        1 prediction column for Numerai Classic.
        At least 1 prediction column and 1 ticker column for Numerai Signals.
        *args, **kwargs are passed to numerapi API.
        For example `version` argument in Numerai Classic submissions.
        """
        self.save_csv(dataf=dataf, file_name=file_name, cols=cols)
        self.upload_predictions(
            file_name=file_name, model_name=model_name,
            *args, **kwargs
        )

    def combine_csvs(self, csv_paths: list,
                     aux_cols: list,
                     era_col: str = None,
                     pred_col: str = 'prediction') -> pd.DataFrame:
        """
        Read in csv files and combine all predictions with a rank mean. \n
        Multi-target predictions will be averaged out. \n
        :param csv_paths: List of full paths to .csv prediction files. \n
        :param aux_cols: ['id'] for Numerai Classic. \n
        ['ticker', 'last_friday', 'data_type'], for example, with Numerai Signals. \n
        :param era_col: Column indicating era ('era' or 'last_friday'). \n
        Will be used for Grouping the rank mean if given. Skip groupby if no era_col provided. \n
        :param pred_col: 'prediction' for Numerai Classic and 'signal' for Numerai Signals.
        """
        all_datafs = [pd.read_csv(path, index_col=aux_cols) for path in tqdm(csv_paths)]
        final_dataf = pd.concat(all_datafs, axis="columns")
        # Remove issue of duplicate columns
        numeric_cols = final_dataf.select_dtypes(include=np.number).columns
        final_dataf.rename({k: str(v) for k, v in zip(numeric_cols, range(len(numeric_cols)))},
                           axis=1,
                           inplace=True)
        # Combine all numeric columns with rank mean
        num_dataf = final_dataf.select_dtypes(include=np.number)
        num_dataf = num_dataf.groupby(era_col) if era_col else num_dataf
        final_dataf[pred_col] = num_dataf.rank(pct=True, method="first").mean(axis=1)
        return final_dataf[[pred_col]]

    def _get_model_id(self, model_name: str) -> str:
        """
        Get ID needed for prediction uploading.
        :param model_name: Raw lowercase model name
        of Numerai model that you have access to.
        """
        return self.get_model_mapping[model_name]

    @property
    def get_model_mapping(self) -> dict:
        """Mapping between raw model names and model IDs."""
        return self.api.get_models()

    def _check_value_range(self, dataf: pd.DataFrame, cols: Union[str, list]):
        """ Check if all predictions are in range (0...1). """
        cols = [cols] if isinstance(cols, str) else cols
        for col in cols:
            if not dataf[col].between(0, 1).all():
                min_val, max_val = dataf[col].min(), dataf[col].max()
                raise ValueError(
                    f"Values must be between 0 and 1. \
Found min value of '{min_val}' and max value of '{max_val}' for column '{col}'."
                )

    def __call__(
            self,
            dataf: pd.DataFrame,
            model_name: str,
            file_name: str = "submission.csv",
            cols: Union[str, list] = "prediction",
            *args,
            **kwargs,
    ):
        """
        The most common use case will be to create a CSV and submit it immediately after that.
        full_submission handles this.
        """
        self.full_submission(
            dataf=dataf,
            file_name=file_name,
            model_name=model_name,
            cols=cols,
            *args,
            **kwargs,
        )


class NumeraiClassicSubmitter(BaseSubmitter):
    """
    Submit for Numerai Classic.

    :param directory_path: Base directory to save and read prediction files from. \n
    :param key: Key object containing valid credentials for Numerai Classic. \n
    *args, **kwargs will be passed to NumerAPI initialization.
    """
    def __init__(self, directory_path: str, key: Key, *args, **kwargs):
        api = NumerAPI(public_id=key.pub_id, secret_key=key.secret_key, *args, **kwargs)
        super().__init__(
            directory_path=directory_path, api=api
        )

    def save_csv(
            self,
            dataf: pd.DataFrame,
            file_name: str = "submission.csv",
            cols: str = 'prediction',
            *args,
            **kwargs,
    ):
        """
        :param dataf: DataFrame which should have at least the following columns:
        1. id (as index column)
        2. cols (for example, 'prediction_mymodel'). Will be saved in 'prediction' column
        :param file_name: .csv file path.
        :param cols: Prediction column name.
        For example, 'prediction' or 'prediction_mymodel'.
        """
        sub_dataf = deepcopy(dataf)
        self._check_value_range(dataf=sub_dataf, cols=cols)

        full_path = str(self.dir / file_name)
        rich_print(
            f":page_facing_up: Saving predictions CSV to '{full_path}'. :page_facing_up:"
        )
        sub_dataf.loc[:, 'prediction'] = sub_dataf[cols]
        sub_dataf.loc[:, 'prediction'].to_csv(full_path, *args, **kwargs)

# %% ../nbs/09_submission.ipynb 26
class NumeraiSignalsSubmitter(BaseSubmitter):
    """
    Submit for Numerai Signals.

    :param directory_path: Base directory to save and read prediction files from. \n
    :param key: Key object containing valid credentials for Numerai Signals. \n
    *args, **kwargs will be passed to SignalsAPI initialization.
    """

    def __init__(self, directory_path: str, key: Key, *args, **kwargs):
        api = SignalsAPI(
            public_id=key.pub_id, secret_key=key.secret_key, *args, **kwargs
        )
        super().__init__(
            directory_path=directory_path, api=api
        )
        self.supported_ticker_formats = [
            "cusip",
            "sedol",
            "ticker",
            "numerai_ticker",
            "bloomberg_ticker",
        ]

    def save_csv(
            self,
            dataf: pd.DataFrame,
            cols: list,
            file_name: str = "submission.csv",
            *args, **kwargs
    ):
        """
        :param dataf: DataFrame which should have at least the following columns:
         1. One of supported ticker formats (cusip, sedol, ticker, numerai_ticker or bloomberg_ticker)
         2. signal (Values between 0 and 1 (exclusive))
         Additional columns for if you include validation data (optional):
         3. friday_date (YYYYMMDD format date indication)
         4. data_type ('val' and 'live' partitions)

         :param cols: All cols that are saved in CSV.
         cols should contain at least 1 ticker column and a 'signal' column.
         For example: ['bloomberg_ticker', 'signal']
         :param file_name: .csv file path.
        """
        self._check_ticker_format(cols=cols)
        self._check_value_range(dataf=dataf, cols="signal")

        full_path = str(self.dir / file_name)
        rich_print(
            f":page_facing_up: Saving Signals predictions CSV to '{full_path}'. :page_facing_up:"
        )
        dataf.loc[:, cols].reset_index(drop=True).to_csv(
            full_path, index=False, *args, **kwargs
        )

    def _check_ticker_format(self, cols: list):
        """ Check for valid ticker format. """
        valid_tickers = set(cols).intersection(set(self.supported_ticker_formats))
        if not valid_tickers:
            raise NotImplementedError(
                f"No supported ticker format in {cols}). \
Supported: '{self.supported_ticker_formats}'"
            )


class NumerBaySubmitter(BaseSubmitter):
    """
    Submit to NumerBay to fulfill sale orders, in addition to submission to Numerai.

    :param tournament_submitter: Base tournament submitter (NumeraiClassicSubmitter or NumeraiSignalsSubmitter). This submitter will use the same directory path.
    :param upload_to_numerai: Whether to also submit to Numerai using the tournament submitter. Defaults to True, set to False to only upload to NumerBay.
    :param numerbay_username: NumerBay username
    :param numerbay_password: NumerBay password
    """
    def __init__(self,
                 tournament_submitter: Union[NumeraiClassicSubmitter, NumeraiSignalsSubmitter],
                 upload_to_numerai: bool = True,
                 numerbay_username: str = None,
                 numerbay_password: str = None):
        super().__init__(
            directory_path=str(tournament_submitter.dir), api=tournament_submitter.api
        )
        from numerbay import NumerBay
        self.numerbay_api = NumerBay(username=numerbay_username, password=numerbay_password)
        self.tournament_submitter = tournament_submitter
        self.upload_to_numerai = upload_to_numerai

    def upload_predictions(self,
                           file_name: str,
                           model_name: str,
                           numerbay_product_full_name: str,
                           *args,
                           **kwargs):
        """
        Upload CSV file to NumerBay (and Numerai if 'upload_to_numerai' is True) for given model name and NumerBay product full name.
        :param file_name: File name/path relative to directory_path.
        :param model_name: Lowercase raw model name (For example, 'integration_test').
        :param numerbay_product_full_name: NumerBay product full name in the format of [category]-[product name], e.g. 'numerai-predictions-numerbay'
        """
        if self.upload_to_numerai:
            self.tournament_submitter.upload_predictions(file_name, model_name, *args, **kwargs)


        full_path = str(self.dir / file_name)
        api_type = str(self.numerbay_api.__class__.__name__)
        rich_print(
            f":airplane: {api_type}: Uploading predictions from '{full_path}' for NumerBay product [bold blue]'{numerbay_product_full_name}'[/bold blue] :airplane:"
        )
        artifact = self.numerbay_api.upload_artifact(
            str(full_path), product_full_name=numerbay_product_full_name
        )
        if artifact:
            rich_print(
                f":thumbs_up: {api_type} submission of '{full_path}' for NumerBay product [bold blue]{numerbay_product_full_name}[/bold blue] is successful! :thumbs_up:"
            )
        else:
            rich_print(f":warning: [red] Upload skipped for NumerBay product '{numerbay_product_full_name}', the product uses buyer-side encryption "
                        "but does not have any active sale order to upload for. [/red] :warning:")

    def full_submission(
        self,
        dataf: pd.DataFrame,
        model_name: str,
        cols: Union[str, list],
        numerbay_product_full_name: str,
        file_name: str = 'submission.csv',
        *args,
        **kwargs,
    ):
        """
        Save DataFrame to csv and upload predictions through API.

        :param dataf: Main DataFrame containing `cols`.
        :param model_name: Lowercase Numerai model name.
        :param numerbay_product_full_name: NumerBay product full name in the format of [category]-[product name], e.g. 'numerai-predictions-numerbay'
        :param file_name: path to save model to relative to base directory.
        :param cols: Columns to be saved in submission file.
        1 prediction column for Numerai Classic.
        At least 1 prediction column and 1 ticker column for Numerai Signals.
        *args, **kwargs are passed to numerapi API.
        For example `version` argument in Numerai Classic submissions.
        """
        self.save_csv(dataf=dataf, file_name=file_name, cols=cols)
        self.upload_predictions(
            file_name=file_name, model_name=model_name, numerbay_product_full_name=numerbay_product_full_name,
            *args, **kwargs
        )

    def combine_csvs(self, *args,**kwargs) -> pd.DataFrame:
        return self.tournament_submitter.combine_csvs(*args,**kwargs)

    def save_csv(self, *args, **kwargs):
        self.tournament_submitter.save_csv(*args, **kwargs)

    @property
    def get_model_mapping(self) -> dict:
        return self.tournament_submitter.api.get_models()

    def __call__(
            self,
            dataf: pd.DataFrame,
            model_name: str,
            numerbay_product_full_name: str,
            file_name: str = "submission.csv",
            cols: Union[str, list] = "prediction",
            *args,
            **kwargs,
    ):
        """
        The most common use case will be to create a CSV and submit it immediately after that.
        full_submission handles this.
        """
        self.full_submission(
            dataf=dataf,
            file_name=file_name,
            model_name=model_name,
            numerbay_product_full_name=numerbay_product_full_name,
            cols=cols,
            *args,
            **kwargs,
        )

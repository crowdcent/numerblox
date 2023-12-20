import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Union, Tuple, Any, List
from numerai_era_data.date_utils import (ERA_ONE_START, get_current_era, 
                                         get_current_date, get_era_for_date,
                                         get_date_for_era)

from .misc import AttrDict
from .feature_groups import (V4_2_FEATURE_GROUP_MAPPING, FNCV3_FEATURES, 
                             SMALL_FEATURES, MEDIUM_FEATURES, V2_EQUIVALENT_FEATURES, 
                             V3_EQUIVALENT_FEATURES)


ERA1_TIMESTAMP = pd.Timestamp(ERA_ONE_START)

class NumerFrame(pd.DataFrame):
    """
    Data structure which extends Pandas DataFrames and
    allows for additional Numerai specific functionality.
    """
    _metadata = ["meta", "feature_cols", "target_cols",
                 "prediction_cols", "not_aux_cols", "aux_cols"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = AttrDict()
        self.__set_era_col()
        self.__init_meta_attrs()
        
    @property
    def _constructor(self):
        return NumerFrame

    def __init_meta_attrs(self):
        """ Dynamically track column groups. """
        self.feature_cols = [col for col in self.columns if str(col).startswith("feature")]
        self.target_cols = [col for col in self.columns if str(col).startswith("target")]
        self.prediction_cols = [
            col for col in self.columns if str(col).startswith("prediction")
        ]
        self.not_aux_cols = self.feature_cols + self.target_cols + self.prediction_cols
        self.aux_cols = [
            col for col in self.columns if col not in self.not_aux_cols
        ]

    def __set_era_col(self):
        """ Each NumerFrame should have an era column to benefit from all functionality. """
        if "era" in self.columns:
            self.meta.era_col = "era"
        elif "friday_date" in self.columns:
            self.meta.era_col = "friday_date"
        elif "date" in self.columns:
            self.meta.era_col = "date"
        else:
            self.meta.era_col = None

    def get_column_selection(self, cols: Union[str, list]) -> "NumerFrame":
        """ Return NumerFrame from selection of columns. """
        return self.loc[:, cols if isinstance(cols, list) else [cols]]

    @property
    def get_feature_data(self) -> "NumerFrame":
        """ All columns for which name starts with 'target'."""
        return self.get_column_selection(cols=self.feature_cols)

    @property
    def get_target_data(self) -> "NumerFrame":
        """ All columns for which name starts with 'target'."""
        return self.get_column_selection(cols=self.target_cols)

    @property
    def get_single_target_data(self) -> "NumerFrame":
        """ Column with name 'target' (Main Numerai target column). """
        return self.get_column_selection(cols=['target'])

    @property
    def get_prediction_data(self) -> "NumerFrame":
        """ All columns for which name starts with 'prediction'."""
        return self.get_column_selection(cols=self.prediction_cols)

    @property
    def get_aux_data(self) -> "NumerFrame":
        """ All columns that are not features, targets or predictions. """
        return self.get_column_selection(cols=self.aux_cols)
    
    @property
    def get_era_data(self) -> "NumerFrame":
        """ Column of all eras. """
        return self.get_column_selection(cols=self.meta.era_col)

    @property
    def get_prediction_aux_data(self) -> "NumerFrame":
        """ All predictions columns and aux columns (for ensembling, etc.). """
        return self.get_column_selection(cols=self.prediction_cols+self.aux_cols)
    
    @property
    def get_fncv3_feature_data(self) -> "NumerFrame":
        """ Get FNCv3 features. """
        return self.get_column_selection(cols=FNCV3_FEATURES)
    
    @property
    def get_small_feature_data(self) -> "NumerFrame":
        """ Small subset of the Numerai dataset for v4.2 data. """
        return self.get_column_selection(cols=SMALL_FEATURES)
    
    @property
    def get_medium_feature_data(self) -> "NumerFrame":
        """ Medium subset of the Numerai dataset for v4.2 data. """
        return self.get_column_selection(cols=MEDIUM_FEATURES)
    
    @property
    def get_v2_equivalent_feature_data(self) -> "NumerFrame":
        """ Features equivalent to the deprecated v2 Numerai data. For v4.2 data. """
        return self.get_column_selection(cols=V2_EQUIVALENT_FEATURES)
    
    @property
    def get_v3_equivalent_feature_data(self) -> "NumerFrame":
        """ Features equivalent to the deprecated v3 Numerai data. For v4.2 data. """
        return self.get_column_selection(cols=V3_EQUIVALENT_FEATURES)

    @property
    def get_unique_eras(self) -> List[str]:
        """ Get all unique eras in the data. """
        return self[self.meta.era_col].unique().tolist()
    
    def get_last_n_eras(self, n: int) -> "NumerFrame":
        """ 
        Get data for the last n eras. 
        Make sure eras are sorted in the way you prefer.
        :param n: Number of eras to select.
        :return: NumerFrame with last n eras.
        """
        eras = self[self.meta.era_col].unique()[-n:]
        return self.loc[self[self.meta.era_col].isin(eras)]
    
    def get_feature_group(self, group: str) -> "NumerFrame":
        """ Get feature group based on name or list of names. """
        assert group in V4_2_FEATURE_GROUP_MAPPING.keys(), \
            f"Group '{group}' not found in {V4_2_FEATURE_GROUP_MAPPING.keys()}"
        return self.get_column_selection(cols=V4_2_FEATURE_GROUP_MAPPING[group])

    def get_pattern_data(self, pattern: str) -> "NumerFrame":
        """
        Get columns based on pattern (for example '_20' to get all 20-day Numerai targets).
        :param pattern: A 'like' pattern (pattern in column_name == True)
        """
        return self.filter(like=pattern)

    def get_feature_target_pair(self, multi_target=False) -> Tuple["NumerFrame", "NumerFrame"]:
        """
        Get split of feature and target columns.
        :param multi_target: Returns only 'target' column by default.
        Returns all target columns when set to True.
        """
        X = self.get_feature_data
        y = self.get_target_data if multi_target else self.get_single_target_data
        return X, y

    def get_era_batch(self, eras: List[Any],
                      convert_to_tf = False,
                      aemlp_batch = False,
                      features: list = None,
                      targets: list = None,
                      *args, **kwargs) -> Tuple["NumerFrame", "NumerFrame"]:
        """
        Get feature target pair batch of 1 or multiple eras. \n
        :param eras: Selection of era names that should be present in era_col. \n
        :param convert_to_tf: Convert to tf.Tensor. \n
        :param aemlp_batch: Specific target batch for autoencoder training. \n
        `y` output will contain three components: features, targets and targets. \n
        :param features: List of features to select. All by default \n
        :param targets: List of targets to select. All by default. \n
        *args, **kwargs are passed to initialization of Tensor.
        """
        valid_eras = []
        for era in eras:
            assert era in self[self.meta.era_col].unique(), f"Era '{era}' not found in era column ({self.meta.era_col})"
            valid_eras.append(era)
        features = features if features else self.feature_cols
        targets = targets if targets else self.target_cols
        X = self.loc[self[self.meta.era_col].isin(valid_eras)][features].values
        y = self.loc[self[self.meta.era_col].isin(valid_eras)][targets].values
        if aemlp_batch:
            y = [X.copy(), y.copy(), y.copy()]

        if convert_to_tf:
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError("TensorFlow is not installed. Please make sure to have Tensorflow installed when setting `convert_to_tf=True`.")
            X = tf.convert_to_tensor(X, *args, **kwargs)
            if aemlp_batch:
                y = [tf.convert_to_tensor(i, *args, **kwargs) for i in y]
            else:
                y = tf.convert_to_tensor(y, *args, **kwargs)
        return X, y
    
    @property
    def get_dates_from_era_col(self) -> pd.Series:
        """ Column of all dates from era column. """
        assert self.meta.era_col == "era", \
            "Era col is not 'era'. Please make sure to have a valid 'era' column to use for converting to dates."
        return self[self.meta.era_col].astype(int).apply(self.get_date_from_era)
    
    @property
    def get_eras_from_date_col(self) -> pd.Series:
        """ Column of all eras from date column. """
        assert self.meta.era_col == "date" or self.meta.era_col == "friday_date", \
            "Era col is not 'date' or 'friday_date'. Please make sure to have a valid 'date' or 'friday_date column to use for converting to eras."
        return self[self.meta.era_col].apply(self.get_era_from_date)
    
    def get_era_range(self, start_era: int, end_era: int) -> "NumerFrame":
        """ 
        Get all eras between two era numbers. 
        :param start_era: Era number to start from (inclusive).
        :param end_era: Era number to end with (inclusive).
        :return: NumerFrame with all eras between start_era and end_era.
        """
        assert "era" in self.columns, "Era column not found. Please make sure to have an 'era' column in your data."
        assert isinstance(start_era, int), f"start_era should be of type 'int' but is '{type(start_era)}'"
        assert isinstance(end_era, int), f"end_era should be of type 'int' but is '{type(end_era)}'"
        assert 1 <= start_era <= end_era <= get_current_era(), \
            f"start_era should be between 1 and {get_current_era()}. Got '{start_era}'."
        assert 1 <= start_era <= end_era <= get_current_era(), \
            f"end_era should be between 1 and {get_current_era()}. Got '{end_era}'."
        assert start_era <= end_era, f"start_era should be before end_era. Got '{start_era}' and '{end_era}'"

        temp_df = self.copy()
        temp_df['era_int'] = temp_df['era'].astype(int)
        result_df = temp_df[(temp_df['era_int'] >= start_era) & (temp_df['era_int'] <= end_era)]
        return result_df.drop(columns=['era_int'])
        
    def get_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> "NumerFrame":
        """
        Get all eras between two dates.
        :param start_date: Starting date (inclusive).
        :param end_date: Ending date (inclusive).
        :return: NumerFrame with all eras between start_date and end_date.
        """
        assert self.meta.era_col == "date" or self.meta.era_col == "friday_date", \
            "Era col is not 'date' or 'friday_date'. Please make sure to have a valid 'era' column."
        assert isinstance(start_date, pd.Timestamp), f"start_date should be of type 'pd.Timestamp' but is '{type(start_date)}'"
        assert isinstance(end_date, pd.Timestamp), f"end_date should be of type 'pd.Timestamp' but is '{type(end_date)}'"
        assert ERA1_TIMESTAMP <= start_date <= pd.Timestamp(get_current_date()), \
            f"start_date should be between {ERA_ONE_START} and {pd.Timestamp(get_current_date())}"
        assert ERA1_TIMESTAMP <= end_date <= pd.Timestamp(get_current_date()), \
            f"end_date should be between {ERA_ONE_START} and {pd.Timestamp(get_current_date())}"
        assert start_date <= end_date, f"start_date should be before end_date. Got '{start_date}' and '{end_date}'"

        temp_df = self.copy()
        result_df = temp_df[(temp_df[self.meta.era_col] >= start_date) & (temp_df[self.meta.era_col] <= end_date)]
        return result_df
    
    @staticmethod
    def get_era_from_date(date_object: pd.Timestamp) -> int:
        """ 
        Get the era number from a specific date. 
        :param date_object: Pandas Timestamp object for which to get era.
        :return: Era number.
        """
        assert isinstance(date_object, pd.Timestamp), f"date_object should be of type 'date' but is '{type(date_object)}'"
        current_date = pd.Timestamp(get_current_date())
        assert ERA1_TIMESTAMP <= date_object <= current_date, \
            f"date_object should be between {ERA_ONE_START} and {current_date}"
        return get_era_for_date(date_object.date())
    
    @staticmethod
    def get_date_from_era(era: int) -> pd.Timestamp:
        """ 
        Get the date from a specific era. 
        :param era: Era number for which to get date.
        Should be an integer which is at least 1.
        :return: Datetime object representing the date of the given era.
        """
        assert isinstance(era, int), f"era should be of type 'int' but is '{type(era)}'"
        assert 1 <= era <= get_current_era(), \
            f"era should be between 1 and {get_current_era()}. Got '{era}'."
        return pd.Timestamp(get_date_for_era(era))



def create_numerframe(file_path: str, columns: list = None, *args, **kwargs) -> NumerFrame:
    """
    Convenient function to initialize NumerFrame.
    Support most used file formats for Pandas DataFrames \n
    (.csv, .parquet, .xls, .pkl, etc.).
    For more details check https://pandas.pydata.org/docs/reference/io.html

    :param file_path: Relative or absolute path to data file. \n
    :param columns: Which columns to read (All by default). \n
    *args, **kwargs will be passed to Pandas loading function.
    """
    assert Path(file_path).is_file(), f"{file_path} does not point to file."
    suffix = Path(file_path).suffix
    if suffix in [".csv"]:
        df = pd.read_csv(file_path, usecols=columns, *args, **kwargs)
    elif suffix in [".parquet"]:
        df = pd.read_parquet(file_path, columns=columns, *args, **kwargs)
    elif suffix in [".xls", ".xlsx", ".xlsm", "xlsb", ".odf", ".ods", ".odt"]:
        df = pd.read_excel(file_path, usecols=columns, *args, **kwargs)
    elif suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(file_path, *args, **kwargs)
        df = df.loc[:, columns] if columns else df
    else:
        raise NotImplementedError(f"Suffix '{suffix}' is not supported.")
    num_frame = NumerFrame(df)
    return num_frame

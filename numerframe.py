import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Any, List

from .misc import AttrDict
from .feature_groups import (V4_2_FEATURE_GROUP_MAPPING, FNCV3_FEATURES, 
                             SMALL_FEATURES, MEDIUM_FEATURES, V2_EQUIVALENT_FEATURES, 
                             V3_EQUIVALENT_FEATURES)


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

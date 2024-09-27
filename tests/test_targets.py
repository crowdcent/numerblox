import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from numerblox.targets import BayesianGMMTargetProcessor, SignalsTargetProcessor

from utils import create_signals_sample_data

dataset = pd.read_parquet("tests/test_assets/val_3_eras.parquet")
dummy_signals_data = create_signals_sample_data

ALL_PROCESSORS = [BayesianGMMTargetProcessor, SignalsTargetProcessor]

def test_processors_sklearn():
    data = dataset.sample(50)
    data = data.drop(columns=["data_type"])
    y = data["target_xerxes_20"].fillna(0.5)
    feature_names = ['feature_melismatic_daily_freak',
                     'feature_pleasurable_facultative_benzol',]
    X = data[feature_names].fillna(0.5)

    for processor_cls in tqdm(ALL_PROCESSORS, desc="Testing target processors for scikit-learn compatibility"):
        # Initialization
        processor = processor_cls()

        # Inherits from Sklearn classes
        assert issubclass(processor_cls, (BaseEstimator, TransformerMixin))

        # Test every processor has get_feature_names_out
        assert hasattr(processor, 'get_feature_names_out'), "Processor {processor.__name__} does not have get_feature_names_out. Every implemented preprocessors should have this method."

def test_bayesian_gmm_target_preprocessor():
    bgmm = BayesianGMMTargetProcessor(n_components=2)

    y = dataset["target_xerxes_20"].fillna(0.5)
    era_series = dataset["era"]
    feature_names = ['feature_melismatic_daily_freak',
                     'feature_pleasurable_facultative_benzol',]
    X = dataset[feature_names]

    bgmm.fit(X, y, era_series=era_series)

    result = bgmm.transform(X, era_series=era_series)
    assert bgmm.get_feature_names_out() == ["fake_target"]
    assert len(result) == len(dataset)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

    # _get_coefs
    coefs = bgmm._get_coefs(X, y, era_series=era_series)
    assert coefs.shape == (3, 2)
    assert coefs.min() >= 0.0
    assert coefs.max() <= 1.0

    # Test set_output API
    bgmm.set_output(transform="pandas")
    result = bgmm.transform(X, era_series=era_series)
    assert isinstance(result, pd.DataFrame)
    bgmm.set_output(transform="default")
    result = bgmm.transform(X, era_series=era_series)
    assert isinstance(result, np.ndarray)

def test_signals_target_processor(dummy_signals_data):
    stp = SignalsTargetProcessor()
    stp.set_output(transform="pandas")
    era_series = dummy_signals_data["date"]
    stp.fit(dummy_signals_data)
    result = stp.transform(dummy_signals_data, era_series=era_series)
    expected_target_cols = ["target_10d_raw", "target_10d_rank", "target_10d_group", "target_20d_raw", "target_20d_rank", "target_20d_group"]
    for col in expected_target_cols:
        assert col in result.columns
    assert stp.get_feature_names_out() == expected_target_cols

    # Test set_output API
    stp.set_output(transform="default")
    result = stp.transform(dummy_signals_data, era_series=era_series)
    assert isinstance(result, np.ndarray)

    stp.set_output(transform="polars")
    result = stp.transform(dummy_signals_data, era_series=era_series)
    assert isinstance(result, pl.DataFrame)
    
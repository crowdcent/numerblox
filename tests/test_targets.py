import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from numerblox.targets import BayesianGMMTargetProcessor, SignalsTargetProcessor

from utils import create_signals_sample_data

dataset = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")
dummy_signals_data = create_signals_sample_data

ALL_PROCESSORS = [BayesianGMMTargetProcessor, SignalsTargetProcessor]

def test_processors_sklearn():
    data = dataset.sample(50)
    data = data.drop(columns=["data_type"])
    y = data["target_jerome_v4_20"].fillna(0.5)
    eras = data["era"]
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    X = data[feature_names].fillna(0.5)

    for processor_cls in tqdm(ALL_PROCESSORS, desc="Testing target processors for scikit-learn compatibility"):
        # Initialization
        processor = processor_cls()

        # Test fit returns self
        try:
            assert processor.fit(X=X, y=y, eras=eras) == processor
        except TypeError:
            assert processor.fit(X=X, y=y) == processor

        # Inherits from Sklearn classes
        assert issubclass(processor_cls, (BaseEstimator, TransformerMixin))

        # Pipeline
        pipeline = Pipeline([
                ('processor', processor),
            ])
        try:
            _ = pipeline.fit(X, y=y, processor__eras=eras)
        except TypeError:
            _ = pipeline.fit(X, y=y)

        # Test every processor has get_feature_names_out
        assert hasattr(processor, 'get_feature_names_out'), "Processor {processor.__name__} does not have get_feature_names_out. Every implemented preprocessors should have this method."

def test_bayesian_gmm_target_preprocessor():
    bgmm = BayesianGMMTargetProcessor(n_components=2)

    y = dataset["target_jerome_v4_20"].fillna(0.5)
    eras = dataset["era"]
    feature_names = ["feature_tallish_grimier_tumbrel",
                     "feature_partitive_labyrinthine_sard"]
    X = dataset[feature_names]

    bgmm.fit(X, y, eras=eras)

    result = bgmm.transform(X, eras=eras)
    assert bgmm.get_feature_names_out() == ["fake_target"]
    assert len(result) == len(dataset)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

    # _get_coefs
    coefs = bgmm._get_coefs(X, y, eras=eras)
    assert coefs.shape == (5, 2)
    assert coefs.min() >= 0.0
    assert coefs.max() <= 1.0

    # Test set_output API
    bgmm.set_output(transform="pandas")
    result = bgmm.transform(X, eras=eras)
    assert isinstance(result, pd.DataFrame)
    bgmm.set_output(transform="default")
    result = bgmm.transform(X, eras=eras)
    assert isinstance(result, np.ndarray)

def test_signals_target_processor(dummy_signals_data):
    stp = SignalsTargetProcessor()
    stp.set_output(transform="pandas")
    eras = dummy_signals_data["date"]
    stp.fit(dummy_signals_data)
    result = stp.transform(dummy_signals_data, eras=eras)
    expected_target_cols = ["target_10d_raw", "target_10d_rank", "target_10d_group", "target_20d_raw", "target_20d_rank", "target_20d_group"]
    for col in expected_target_cols:
        assert col in result.columns
    assert stp.get_feature_names_out() == expected_target_cols

    # Test set_output API
    stp.set_output(transform="default")
    result = stp.transform(dummy_signals_data, eras=eras)
    assert isinstance(result, np.ndarray)
import pytest
import pandas as pd
import numpy as np

from numerblox.numerframe import NumerFrame
from numerblox.neutralizers import BaseNeutralizer, FeatureNeutralizer, FeaturePenalizer

@pytest.fixture
def setup_data():
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": [4, 3, 2, 1],
        "prediction": [0.5, 0.6, 0.7, 0.8],
        "era": ["era1", "era2", "era1", "era2"]
    }
    return NumerFrame(data)


def test_base_neutralizer_initialization():
    bn = BaseNeutralizer(new_col_name="test")
    assert bn.new_col_name == "test"

def test_base_neutralizer_fit(setup_data):
    obj = BaseNeutralizer(new_col_name="test").fit(setup_data)
    assert isinstance(obj, BaseNeutralizer)

def test_feature_neutralizer_initialization():
    fn = FeatureNeutralizer()
    assert fn.new_col_name.startswith("prediction_neutralized_")
    assert FeatureNeutralizer(feature_names=["feature1", "feature2"]).feature_names == ["feature1", "feature2"]

def test_feature_penalizer_initialization():
    fp = FeaturePenalizer(feature_names=["feature1", "feature2"], max_exposure=0.5)
    assert fp.new_col_name.startswith("prediction_penalized_")
    assert fp.max_exposure == 0.5
    assert fp.feature_names == ["feature1", "feature2"]

def test_feature_neutralizer_predict(setup_data):
    fn = FeatureNeutralizer()
    result = fn.predict(setup_data)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_feature_neutralizer_get_feature_names_out():
    names = FeatureNeutralizer(feature_names=["feature1", "feature2"]).get_feature_names_out()
    assert names == ["prediction_neutralized_0.5"]

def test_feature_neutralizer_get_feature_names_out_with_input_features():
    names = FeatureNeutralizer(feature_names=["feature1", "feature2"]).get_feature_names_out(input_features=["prediction_fancy1"])
    assert names == ["prediction_fancy1"]

def test_feature_neutralizer_neutralize(setup_data):
    columns = ["prediction"]
    by = ["feature1", "feature2"]
    scores = FeatureNeutralizer(feature_names=["feature1", "feature2"]).neutralize(setup_data, columns, by)
    assert isinstance(scores, pd.DataFrame)

# TODO Fast FeaturePenalizer tests
# def test_feature_penalizer_predict(setup_data):
#     fp = FeaturePenalizer(max_exposure=0.5)
#     result = fp.predict(setup_data)
#     assert len(result) == len(setup_data)
#     assert result['prediction'].min() >= 0
#     assert result['prediction'].max() <= 1

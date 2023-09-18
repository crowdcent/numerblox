import pytest
import numpy as np 
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from numerblox.meta import make_numerai_pipeline, make_numerai_union
from numerblox.neutralizers import BaseNeutralizer, FeatureNeutralizer, FeaturePenalizer

@pytest.fixture
def setup_data():
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": [4, 3, 2, 1],
        "prediction": [0.5, 0.6, 0.7, 0.8],
        "target": [0, 1, 0, 1],
        "era": ["era1", "era2", "era1", "era2"]
    }
    return pd.DataFrame(data)


def test_base_neutralizer_initialization():
    bn = BaseNeutralizer(new_col_name="test")
    assert bn.new_col_name == "test"

def test_base_neutralizer_fit(setup_data):
    obj = BaseNeutralizer(new_col_name="test").fit(setup_data)
    assert isinstance(obj, BaseNeutralizer)

def test_feature_neutralizer_initialization():
    fn = FeatureNeutralizer()
    assert fn.new_col_name.startswith("prediction_neutralized_")

def test_feature_penalizer_initialization():
    fp = FeaturePenalizer(max_exposure=0.5)
    assert fp.new_col_name.startswith("prediction_penalized_")
    assert fp.max_exposure == 0.5

def test_feature_neutralizer_predict(setup_data):
    fn = FeatureNeutralizer()
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"]
    result = fn.predict(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_feature_neutralizer_get_feature_names_out():
    names = FeatureNeutralizer().get_feature_names_out()
    assert names == ["prediction_neutralized_0.5"]

def test_feature_neutralizer_get_feature_names_out_complex():
    names = FeatureNeutralizer(pred_name="fancy", suffix="blob").get_feature_names_out()
    assert names == ["fancy_neutralized_0.5_blob"]

def test_feature_neutralizer_get_feature_names_out_with_input_features():
    names = FeatureNeutralizer().get_feature_names_out(input_features=["prediction_fancy1"])
    assert names == ["prediction_fancy1"]

def test_feature_neutralizer_neutralize(setup_data):
    columns = ["prediction"]
    by = ["feature1", "feature2"]
    scores = FeatureNeutralizer().neutralize(setup_data, columns, by)
    assert isinstance(scores, pd.DataFrame)

def test_feature_neutralizer_pipeline(setup_data):
    lr1 = Ridge()
    fn = FeatureNeutralizer()
    pipeline = make_numerai_pipeline(lr1, fn)
    pipeline.fit(setup_data[["feature1", "feature2"]], setup_data["target"])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]

    result = pipeline.predict(setup_data[["feature1", "feature2"]],
                              features=features, eras=eras)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_feature_neutralizer_featureunion(setup_data):
    onehot = OneHotEncoder(sparse_output=False)
    pipeline = make_numerai_pipeline(Ridge(), FeatureNeutralizer())
    union = make_numerai_union(onehot, pipeline)
    print(union.transformer_list)

    # Fitting the union
    union.fit(setup_data[["feature1", "feature2"]], setup_data["target"])

    # Getting the features and eras
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]

    # Making predictions
    result = union.transform(setup_data[["feature1", "feature2"]], 
                             numeraipipeline={"features": features, "eras": eras})

    # Your assertions
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    # All one hot should be in the right place with right values.
    assert np.all(np.isin(result[:, :-1], [0, 1]))
    # All pipeline predictions should be between 0 and 1.
    assert np.all((0 <= result[:, -1]) & (result[:, -1] <= 1))

# TODO Test for ColumnTransformer
def test_feature_neutralizer_columntransformer(setup_data):
    pass

# TODO Fast FeaturePenalizer tests
# def test_feature_penalizer_predict(setup_data):
#     fp = FeaturePenalizer(max_exposure=0.5)
#     features = setup_data[["feature1", "feature2"]]
#     eras = setup_data["era"]
#     X = setup_data["prediction"]
#     result = fp.predict(X, features=features, eras=eras)
#     assert len(result) == len(setup_data)
#     assert result['prediction'].min() >= 0
#     assert result['prediction'].max() <= 1

# def test_feature_penalizer_pipeline(setup_data):
#     lr1 = Ridge()
#     fp = FeaturePenalizer(max_exposure=0.5)
#     pipeline = make_numerai_pipeline(lr1, fp)
#     pipeline.fit(setup_data[["feature1", "feature2"]], setup_data["target"])
#     features = setup_data[["feature1", "feature2"]]
#     eras = setup_data["era"]

#     result = pipeline.predict(setup_data[["feature1", "feature2"]],
#                               features=features, eras=eras)
#     assert isinstance(result, np.ndarray)
#     assert len(result) == len(setup_data)
#     assert result.min() >= 0
#     assert result.max() <= 1

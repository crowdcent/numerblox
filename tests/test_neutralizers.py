import pandas as pd

from numerblox.neutralizers import BaseNeutralizer, FeatureNeutralizer

from utils import create_classic_sample_data


setup_data = create_classic_sample_data

##### Basic tests #####

def test_base_neutralizer_initialization():
    bn = BaseNeutralizer(new_col_name="test")
    assert bn.new_col_name == "test"

def test_base_neutralizer_fit(setup_data):
    obj = BaseNeutralizer(new_col_name="test").fit(setup_data)
    assert isinstance(obj, BaseNeutralizer)

def test_feature_neutralizer_initialization():
    fn = FeatureNeutralizer()
    assert fn.new_col_name.startswith("prediction_neutralized_")

def test_feature_neutralizer_predict(setup_data):
    fn = FeatureNeutralizer()
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"]
    result = fn.transform(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_feature_neutralizer_neutralize(setup_data):
    columns = ["prediction"]
    by = ["feature1", "feature2"]
    scores = FeatureNeutralizer().neutralize(setup_data, columns, by)
    assert isinstance(scores, pd.DataFrame)

def test_feature_neutralizer_get_feature_names_out():
    names = FeatureNeutralizer().get_feature_names_out()
    assert names == ["prediction_neutralized_0.5"]

def test_feature_neutralizer_get_feature_names_out_complex():
    names = FeatureNeutralizer(pred_name="fancy", suffix="blob").get_feature_names_out()
    assert names == ["fancy_neutralized_0.5_blob"]

def test_feature_neutralizer_get_feature_names_out_with_input_features():
    names = FeatureNeutralizer().get_feature_names_out(input_features=["prediction_fancy1"])
    assert names == ["prediction_fancy1"]

import pandas as pd

from numerblox.pipeline import make_numerai_pipeline, make_numerai_union
from numerblox.neutralizers import BaseNeutralizer, FeatureNeutralizer, FeaturePenalizer

from utils import create_classic_sample_data


setup_data = create_classic_sample_data

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

# TODO Test for ColumnTransformer
def test_feature_neutralizer_columntransformer(setup_data):
    pass



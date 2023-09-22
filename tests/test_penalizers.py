from numerblox.penalizers import BasePenalizer, FeaturePenalizer

from utils import create_classic_sample_data

setup_data = create_classic_sample_data

def test_base_neutralizer_initialization():
    bn = BasePenalizer(new_col_name="test")
    assert bn.new_col_name == "test"

def test_base_neutralizer_fit(setup_data):
    obj = BasePenalizer(new_col_name="test").fit(setup_data)
    assert isinstance(obj, BasePenalizer)

def test_feature_penalizer_initialization():
    fp = FeaturePenalizer(max_exposure=0.5)
    assert fp.new_col_name.startswith("prediction_penalized_")
    assert fp.max_exposure == 0.5

def test_feature_penalizer_get_feature_names_out():
    names = FeaturePenalizer(max_exposure=0.5).get_feature_names_out()
    assert names == ["prediction_penalized_0.5"]

def test_feature_penalizer_get_feature_names_out_complex():
    names = FeaturePenalizer(max_exposure=0.7, pred_name="fancy", suffix="blob").get_feature_names_out()
    assert names == ["fancy_penalized_0.7_blob"]

def test_feature_penalizer_get_feature_names_out_with_input_features():
    names = FeaturePenalizer(max_exposure=0.5).get_feature_names_out(input_features=["prediction_fancy1"])
    assert names == ["prediction_fancy1"]
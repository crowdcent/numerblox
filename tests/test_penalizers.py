from numerblox.penalizers import BasePenalizer, FeaturePenalizer

from utils import create_classic_sample_data

setup_data = create_classic_sample_data

def test_base_penalizer_initialization():
    bn = BasePenalizer(new_col_name="test")
    assert bn.new_col_name == "test"

def test_base_penalizer_fit(setup_data):
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
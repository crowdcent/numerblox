import pytest
import numpy as np
import pandas as pd

from numerblox.neutralizers import BaseNeutralizer, FeatureNeutralizer

from utils import create_classic_sample_data

setup_data = create_classic_sample_data

def test_base_neutralizer_initialization():
    bn = BaseNeutralizer(new_col_names=["test"])
    assert bn.new_col_names == ["test"]

def test_base_neutralizer_fit(setup_data):
    obj = BaseNeutralizer(new_col_names=["test"]).fit(setup_data)
    assert isinstance(obj, BaseNeutralizer)

def test_feature_neutralizer_initialization():
    fn = FeatureNeutralizer()
    assert fn.new_col_names[0].startswith("prediction_neutralized_")

    # Proportion must be between 0 and 1
    with pytest.raises(AssertionError):
        FeatureNeutralizer(proportion=[1.1])
    with pytest.raises(AssertionError):
        FeatureNeutralizer(proportion=[-0.1])

def test_feature_neutralizer_length_mismatch_X_features(setup_data):
    fn = FeatureNeutralizer()
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"][:-1]  # Remove one element to cause mismatch

    with pytest.raises(AssertionError):
        fn.transform(X, features, eras)

def test_feature_neutralizer_length_mismatch_X_eras(setup_data):
    fn = FeatureNeutralizer()
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"][:-1]  # Remove one element to cause mismatch
    X = setup_data["prediction"]

    with pytest.raises(AssertionError):
        fn.transform(X, features, eras)

def test_feature_neutralizer_incorrect_dim_X_single_pred(setup_data):
    fn = FeatureNeutralizer(pred_name=["prediction1", "prediction2"])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"]  # X is 1D, but two prediction names are provided

    with pytest.raises(AssertionError):
        fn.transform(X, features, eras)

def test_feature_neutralizer_incorrect_dim_X_multi_pred(setup_data):
    fn = FeatureNeutralizer(pred_name=["prediction1", "prediction2"])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    setup_data["prediction2"] = np.random.uniform(size=len(setup_data))
    X = setup_data[["prediction"]]  # Only one column provided, but two expected

    with pytest.raises(AssertionError):
        fn.transform(X, features, eras)

def test_feature_neutralizer_predict(setup_data):
    fn = FeatureNeutralizer(pred_name="prediction", proportion=0.5)
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"]
    result = fn.transform(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.shape[1] == 1
    assert np.all(np.isclose(result, 0, atol=1e-8) | (result >= 0))
    assert np.all(np.isclose(result, 1, atol=1e-8) | (result <= 1))

def test_feature_neutralizer_predict_multi_pred(setup_data):
    fn = FeatureNeutralizer(pred_name=["prediction", "prediction2"], proportion=[0.5])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    setup_data["prediction2"] = np.random.uniform(size=len(setup_data))
    X = setup_data[["prediction", "prediction2"]]
    result = fn.transform(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.shape[1] == 2
    assert np.all(np.isclose(result, 0, atol=1e-8) | (result >= 0))
    assert np.all(np.isclose(result, 1, atol=1e-8) | (result <= 1))

def test_feature_neutralizer_predict_multi_prop(setup_data):
    fn = FeatureNeutralizer(pred_name="prediction", proportion=[0.5, 0.7])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    X = setup_data["prediction"]
    result = fn.transform(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.shape[1] == 2
    assert np.all(np.isclose(result, 0, atol=1e-8) | (result >= 0))
    assert np.all(np.isclose(result, 1, atol=1e-8) | (result <= 1))

def test_feature_neutralizer_multi_pred_multi_prop(setup_data):
    fn = FeatureNeutralizer(pred_name=["prediction", "prediction2"], proportion=[0.5, 0.7, 0.9])
    features = setup_data[["feature1", "feature2"]]
    eras = setup_data["era"]
    setup_data["prediction2"] = np.random.uniform(size=len(setup_data))
    X = setup_data[["prediction", "prediction2"]]
    result = fn.transform(X, features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.shape[1] == 6
    assert np.all(np.isclose(result, 0, atol=1e-8) | (result >= 0))
    assert np.all(np.isclose(result, 1, atol=1e-8) | (result <= 1))

    # Test with numpy X
    result = fn.transform(X.to_numpy(), features=features, eras=eras)
    assert len(result) == len(setup_data)
    assert result.shape[1] == 6
    assert np.all(np.isclose(result, 0, atol=1e-8) | (result >= 0))
    assert np.all(np.isclose(result, 1, atol=1e-8) | (result <= 1))

def test_feature_neutralizer_neutralize(setup_data):
    columns = ["prediction"]
    by = ["feature1", "feature2"]
    scores = FeatureNeutralizer().neutralize(setup_data, columns, by, proportion=0.5)
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

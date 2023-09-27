import pytest
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.datasets import make_regression
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from numerblox.ensemble import NumeraiEnsemble, PredictionReducer

##### Mock objects #####
@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)

@pytest.fixture
def ensemble():
    return NumeraiEnsemble()

##### NumeraiEnsemble #####

def test_numeraiensemble_fit(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    check_is_fitted(ensemble)
    assert issubclass(type(ensemble), (BaseEstimator, TransformerMixin))

def test_numeraiensemble_predict(ensemble, sample_data):
    X, y = sample_data
    ensemble = NumeraiEnsemble(weights=[0.05, 0.05, 0.3, 0.3, 0.3])
    ensemble.fit(X, y)
    eras = np.array([1]*50 + [2]*50)
    input_preds = np.random.uniform(size=(100, 5))

    ensemble_preds = ensemble.predict(input_preds, eras)
    # The length of output should have the same shape as input preds
    assert len(ensemble_preds) == len(input_preds)
    # Output should be a numpy array with values between 0 and 1
    assert isinstance(ensemble_preds, np.ndarray) 
    assert len(ensemble_preds.shape) == 2
    assert ensemble_preds.min() >= 0
    assert ensemble_preds.max() <= 1

    # Test with Pandas Series into
    input_preds = pd.DataFrame(input_preds)
    eras = pd.Series(eras)
    ensemble_preds = ensemble.predict(input_preds, eras)
    

def test_numeraiensemble_standardize(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    
    data = np.array([1, 2, 3, 4, 5])
    standardized_data = ensemble._standardize(data)
    
    expected = (rankdata(data, method="ordinal") - 0.5) / len(data)
    
    assert np.allclose(standardized_data, expected)

def test_numeraiensemble_standardize_by_era(ensemble):
    eras = np.array([1, 1, 1, 2, 2, 2])

    # Test 1: Basic functionality
    X = np.array([0.5, 0.7, 0.1, 0.9, 0.6, 0.3])
    standardized = ensemble._standardize_by_era(X, eras)
    # These values are simply computed based on manual calculations for rank and normalization
    expected_values_1 = [0.5, 0.83333333, 0.16666667, 0.83333333, 0.5, 0.16666667]
    assert np.allclose(standardized, expected_values_1)

    # Test 2: Check standardized values for all same predictions split across two different eras
    X = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    standardized = ensemble._standardize_by_era(X, eras)
    expected_values_2 = [0.16666667, 0.5 ,0.83333333, 0.16666667, 0.5, 0.83333333]
    assert np.allclose(standardized, expected_values_2)

    # Test 3: Different predictions but split across two eras
    X = np.array([0.1, 0.9, 0.9, 0.1, 0.1, 0.9])
    standardized = ensemble._standardize_by_era(X, eras)
    expected_values_3 = [0.16666667, 0.5, 0.83333333, 0.16666667, 0.5, 0.83333333]
    assert np.allclose(standardized, expected_values_3)

def test_numeraiensemble_predict_with_constant_values(ensemble):
    # Create an instance of your ensemble with mock estimators
    eras = np.random.randint(1, 5, size=100)
    
    X_fit = np.random.rand(100, 3)
    y_fit = np.random.rand(100)
    ensemble.fit(X_fit, y_fit)

    constant_preds = np.ones((100, 5))

    with pytest.raises(ValueError, match="Predictions for all columns are constant. No valid predictions to ensemble."):
        with pytest.warns(UserWarning, match="Some estimator predictions are constant. Consider checking your estimators. Skipping these estimator predictions in ensembling."):
            ensemble.predict(constant_preds, eras)

def test_numeraiensemble_predict_with_nans(ensemble):
    # Create an instance of your ensemble with mock estimators
    eras = np.random.randint(1, 5, size=100)
    
    X_fit = np.random.rand(100, 3)
    y_fit = np.random.rand(100)
    ensemble.fit(X_fit, y_fit)

    nan_preds = np.ones((100, 5))
    nan_preds[5:15, 0] = np.nan
    nan_preds[:5, 1] = np.nan

    with pytest.warns(UserWarning, match="Predictions in column"):
        ensemble_preds = ensemble.predict(nan_preds, eras)
    assert len(ensemble_preds) == len(nan_preds)
    # Output should be a numpy array with values between 0 and 1
    assert isinstance(ensemble_preds, np.ndarray) 
    # There must be some nans in the data.
    assert np.sum(np.isnan(ensemble_preds)) >= 0
    # None nan values should be between 0 and 1
    non_nan_values = ensemble_preds[~np.isnan(ensemble_preds)]
    if non_nan_values.size > 0:
        assert non_nan_values.min() >= 0
        assert non_nan_values.max() <= 1

def test_numeraiensemble_donate_weights(ensemble):
    ensemble.donate_weighted = True
    # For 3 predictions, weights should be [0.25, 0.25, 0.5]
    assert ensemble._get_donate_weights(n=3) == [0.25, 0.25, 0.5]
    # For 5, weights should be [0.0625, 0.0625, 0.125, 0.25, 0.5]
    assert ensemble._get_donate_weights(n=5) == [0.0625, 0.0625, 0.125, 0.25, 0.5]

def test_numeraiensemble_donate_weights_sum_to_one(ensemble):
    ensemble.donate_weighted = True
    for n_estimators in range(1, 11):
        # Assert that the sum of weights is close to 1
        assert np.isclose(sum(ensemble._get_donate_weights(n=n_estimators)), 1.0)

def test_numeraiensemble_get_feature_names_out(ensemble):
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    ensemble.fit(X, y)
    assert ensemble.get_feature_names_out() == ["numerai_ensemble_predictions"]
    assert ensemble.get_feature_names_out(['a', 'b']) == ['a', 'b']

def test_numeraiensemble_set_output(ensemble, sample_data):
    X, y = sample_data
    eras = np.array([1]*50 + [2]*50)
    ens_ins = ensemble
    ens_ins.fit(X, y)

    ens_ins.set_output(transform="pandas")
    preds = ens_ins.predict(X, eras=eras)
    assert isinstance(preds, pd.DataFrame)
    ens_ins.set_output(transform="default")
    preds = ens_ins.predict(X, eras=eras)
    assert isinstance(preds, np.ndarray)

##### PredictionReducer #####

def test_prediction_reducer():
    # Simulated probability predictions for 3 samples, 2 models and 3 classes
    X = np.array([
        [0.1, 0.7, 0.2,  0.2, 0.5, 0.3],
        [0.2, 0.5, 0.3,  0.3, 0.3, 0.4],
        [0.6, 0.2, 0.2,  0.4, 0.4, 0.2]
    ])
    
    reducer = PredictionReducer(n_models=2, n_classes=3)
    reduced_X = reducer.fit_transform(X)

    # The expected result is a 3x2 matrix
    expected_result = np.array([
        [0.7 * 1 + 0.2 * 2,  0.5 * 1 + 0.3 * 2],
        [0.5 * 1 + 0.3 * 2,  0.3 * 1 + 0.4 * 2],
        [0.2 * 1 + 0.2 * 2,  0.4 * 1 + 0.2 * 2]
    ])

    assert reduced_X.shape == (3, 2)
    np.testing.assert_array_almost_equal(reduced_X, expected_result)

    assert issubclass(type(reducer), (BaseEstimator, TransformerMixin))

    # Set output API
    reducer.set_output(transform="pandas")
    preds = reducer.predict(X)
    assert isinstance(preds, pd.DataFrame)
    reducer.set_output(transform="default")
    preds = reducer.predict(X)
    assert isinstance(preds, np.ndarray)

def test_prediction_reducer_feature_names_out():
    reducer = PredictionReducer(n_models=3, n_classes=4)
    feature_names = reducer.get_feature_names_out()
    expected_names = ["reduced_prediction_0", "reduced_prediction_1", "reduced_prediction_2"]
    
    assert feature_names == expected_names


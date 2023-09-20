

import pytest
import numpy as np
from copy import deepcopy
from scipy.stats import rankdata, norm
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from numerblox.meta import NumeraiEnsemble, MetaEstimator

##### Mock objects #####
@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)

@pytest.fixture
def ensemble():
    return NumeraiEnsemble()

class MockEstimator(BaseEstimator, RegressorMixin):
    """ A mock estimator that always predicts a constant value. """
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.ones(X.shape[0]) * 3
    
class ValidMockEstimator(BaseEstimator, RegressorMixin):
    """A mock estimator that always predicts values within [0, 1] range."""
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.uniform(size=len(X))

class OutOfBoundsMockEstimator(BaseEstimator, RegressorMixin):
    """A mock estimator that always predicts values out of [0, 1] range."""
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.array([1.5] * len(X))

##### NumeraiEnsemble #####

def test_numeraiensemble_fit(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    check_is_fitted(ensemble)
    assert hasattr(ensemble, "get_feature_names_out")

def test_numeraiensemble_predict(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    eras = np.array([1]*50 + [2]*50)
    input_preds = np.random.uniform(size=(100, 5))

    ensemble_preds = ensemble.predict(input_preds, eras)
    # The length of output should have the same shape as input preds
    assert len(ensemble_preds) == len(input_preds)
    # Output should be a numpy array with values between 0 and 1
    assert isinstance(ensemble_preds, np.ndarray) 
    assert ensemble_preds.min() >= 0
    assert ensemble_preds.max() <= 1

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

    with pytest.raises(ValueError, match="Predictions for all estimators are constant. No valid predictions to ensemble."):
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

    ensemble_preds = ensemble.predict(nan_preds, eras)
    assert len(ensemble_preds) == len(nan_preds)
    # Output should be a numpy array with values between 0 and 1
    assert isinstance(ensemble_preds, np.ndarray) 
    assert ensemble_preds.min() >= 0
    assert ensemble_preds.max() <= 1

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

##### MetaEstimator #####

def test_meta_estimator_init():
    with pytest.raises(AssertionError):
        MetaEstimator(ValidMockEstimator(), predict_func="predict_proba")
    with pytest.raises(AssertionError):
        MetaEstimator(ValidMockEstimator(), predict_func="hello")

def test_meta_estimator_multioutput():
    # Create dummy multioutput dataset
    X, y = make_regression(n_samples=100, n_features=20, n_targets=3, noise=0.1)

    # Multioutput model
    model_multi = RandomForestRegressor()
    meta_estimator_multi = MetaEstimator(model_multi)
    meta_estimator_multi.fit(X, y)
    assert meta_estimator_multi.multi_output_ == True
    assert meta_estimator_multi.estimator_.__class__ == model_multi.__class__
    transformed = meta_estimator_multi.transform(X)

    assert transformed.shape == (X.shape[0], y.shape[1]), f"Expected shape {(X.shape[0], y.shape[1])}, but got {transformed.shape}"

def test_meta_estimator_get_feature_names_out():
    ensemble = MetaEstimator(ValidMockEstimator(), predict_func="predict")
    assert ensemble.get_feature_names_out() == ["ValidMockEstimator_predict_predictions"]
    assert ensemble.get_feature_names_out(['a', 'b']) == ['a', 'b']
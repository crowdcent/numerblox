

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import rankdata, norm
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

from numerblox.meta import NumeraiEnsemble

@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)

@pytest.fixture
def ensemble():
    lr1 = LinearRegression()
    lr2 = Ridge()
    # Dummy eras
    eras = []
    for i in range(1, 11):
        eras.extend(["{:04}".format(i)] * 10)
    return NumeraiEnsemble(estimators=[('lr1', lr1), ('lr2', lr2)], 
                           eras=eras, gaussianize=False)

def test_fit(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    check_is_fitted(ensemble)
    assert hasattr(ensemble, "estimators_")
    assert hasattr(ensemble, "get_feature_names_out")

def test_predict(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)

    preds = ensemble.predict(X)
    # The output of predict should have the same shape as y
    assert preds.shape == y.shape
    # Output should be a numpy array with values between 0 and 1
    assert isinstance(preds, np.ndarray)
    assert preds.min() >= 0
    assert preds.max() <= 1

def test_standardize_without_gaussianize(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    
    data = np.array([1, 2, 3, 4, 5])
    standardized_data = ensemble._standardize(data)
    
    expected = (rankdata(data, method="ordinal") - 0.5) / len(data)
    
    assert np.allclose(standardized_data, expected)

def test_standardize_with_gaussianize(ensemble, sample_data):
    X, y = sample_data
    ensemble.gaussianize = True
    ensemble.fit(X, y)
    
    data = np.array([1, 2, 3, 4, 5])
    standardized_data = ensemble._standardize(data)
    
    expected_percentiles = (rankdata(data, method="ordinal") - 0.5) / len(data)
    expected = norm.ppf(expected_percentiles)
    
    assert np.allclose(standardized_data, expected)

def test_standardize_by_era():
    eras = np.array([1, 1, 1, 2, 2, 2])
    ensemble = NumeraiEnsemble(estimators=None, eras=eras,
                               gaussianize=False)

    # Test 1: Basic functionality
    X = np.array([0.5, 0.7, 0.1, 0.9, 0.6, 0.3])
    standardized = ensemble._standardize_by_era(X)
    # These values are simply computed based on manual calculations for rank and normalization
    expected_values_1 = [0.5, 0.83333333, 0.16666667, 0.83333333, 0.5, 0.16666667]
    assert np.allclose(standardized, expected_values_1)

    # Test 2: Check standardized values for all same predictions split across two different eras
    X = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    standardized = ensemble._standardize_by_era(X)
    expected_values_2 = [0.16666667, 0.5 ,0.83333333, 0.16666667, 0.5, 0.83333333]
    assert np.allclose(standardized, expected_values_2)

    # Test 3: Different predictions but split across two eras
    X = np.array([0.1, 0.9, 0.9, 0.1, 0.1, 0.9])
    standardized = ensemble._standardize_by_era(X)
    expected_values_3 = [0.16666667, 0.5, 0.83333333, 0.16666667, 0.5, 0.83333333]
    assert np.allclose(standardized, expected_values_3)


class MockEstimator(BaseEstimator, RegressorMixin):
    """ A mock estimator that always predicts a constant value. """
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.ones(X.shape[0]) * 3

def test_predict_with_constant_values():
    # Create an instance of your ensemble with mock estimators
    estimators = [('mock1', MockEstimator()), ('mock2', MockEstimator())]
    eras = np.random.randint(1, 5, size=10)
    ensemble = NumeraiEnsemble(estimators, eras=eras)
    
    X = np.random.rand(10, 3)
    
    X_fit = np.random.rand(10, 3)
    y_fit = np.random.rand(10)
    ensemble.fit(X_fit, y_fit)

    with pytest.raises(ValueError, match="Predictions for all estimators are constant. No valid predictions to ensemble."):
        ensemble.predict(X)

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

def test_ensembled_predictions_out_of_bounds():
    # Use the mock estimator for the ensemble
    estimators = [('mock1', OutOfBoundsMockEstimator()), ('mock2', ValidMockEstimator())]
    eras = np.random.randint(1, 5, size=10)
    ensemble = NumeraiEnsemble(estimators, eras=eras)
    
    # Fit the ensemble with dummy data
    X_fit = np.random.rand(10, 3)
    y_fit = np.random.rand(10)
    ensemble.fit(X_fit, y_fit)
    
    X = np.random.rand(10, 3)
    
    # The mock estimator will predict values of 1.5, which are out of [0, 1] range
    with pytest.raises(ValueError, match="Ensembled predictions are not between 0 and 1. Consider checking your estimators."):
        ensemble.predict(X)

def test_donate_weights():
    estimators = [('mock1', ValidMockEstimator()), ('mock2', ValidMockEstimator()),
                  ('mock2', ValidMockEstimator())]
    eras = np.random.randint(1, 5, size=10)
    ensemble = NumeraiEnsemble(estimators, eras=eras, donate_weighted=True)

    # For 3 mock estimators, weights should be [0.25, 0.25, 0.5]
    assert ensemble.weights == [0.25, 0.25, 0.5]

    # For 5 mock estimators, weights should be [0.0625, 0.0625, 0.125, 0.25, 0.5]
    estimators = [('mock1', ValidMockEstimator()), ('mock2', ValidMockEstimator()),
                  ('mock3', ValidMockEstimator()), ('mock4', ValidMockEstimator()),
                  ('mock5', ValidMockEstimator())]
    ensemble = NumeraiEnsemble(estimators, eras=eras, donate_weighted=True)
    assert ensemble.weights == [0.0625, 0.0625, 0.125, 0.25, 0.5]

def test_donate_weights_sum_to_one():
    for n_estimators in range(1, 11):
        estimators = [('mock' + str(i), ValidMockEstimator()) for i in range(n_estimators)]
        eras = np.random.randint(1, 5, size=10)
        ensemble = NumeraiEnsemble(estimators, eras=eras, donate_weighted=True)

        # Assert that the sum of weights is close to 1
        assert np.isclose(sum(ensemble.weights), 1.0)

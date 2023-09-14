

import numpy as np
import pytest
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import rankdata, norm
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

from numerblox.numerframe import NumerFrame, create_numerframe
from numerblox.meta import NumeraiEnsemble, NeutralizedEstimator
from numerblox.neutralizers import FeatureNeutralizer

@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)

@pytest.fixture
def ensemble():
    lr1 = LinearRegression()
    lr2 = Ridge()
    return NumeraiEnsemble(estimators=[('lr1', lr1), ('lr2', lr2)], gaussianize=False)

def test_fit(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    check_is_fitted(ensemble)
    assert hasattr(ensemble, "estimators_")
    assert hasattr(ensemble, "get_feature_names_out")

def test_predict(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    # Dummy eras
    eras = []
    for i in range(1, 11):
        eras.extend(["{:04}".format(i)] * 10)

    preds = ensemble.predict(X, eras)
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
    standardized_data = ensemble.standardize(data)
    
    expected = (rankdata(data, method="ordinal") - 0.5) / len(data)
    
    assert np.allclose(standardized_data, expected)

def test_standardize_with_gaussianize(ensemble, sample_data):
    X, y = sample_data
    ensemble.gaussianize = True
    ensemble.fit(X, y)
    
    data = np.array([1, 2, 3, 4, 5])
    standardized_data = ensemble.standardize(data)
    
    expected_percentiles = (rankdata(data, method="ordinal") - 0.5) / len(data)
    expected = norm.ppf(expected_percentiles)
    
    assert np.allclose(standardized_data, expected)

def test_standardize_by_era():
    ensemble = NumeraiEnsemble(estimators=None, gaussianize=False)

    # Test 1: Basic functionality
    X = np.array([0.5, 0.7, 0.1, 0.9, 0.6, 0.3])
    eras = np.array([1, 1, 1, 2, 2, 2])
    standardized = ensemble.standardize_by_era(X, eras)
    # These values are simply computed based on manual calculations for rank and normalization
    expected_values_1 = [0.5, 0.83333333, 0.16666667, 0.83333333, 0.5, 0.16666667]
    assert np.allclose(standardized, expected_values_1)

    # Test 2: Check standardized values for all same predictions split across two different eras
    X = np.array([0.5, 0.5, 0.5, 0.5])
    eras = np.array([1, 1, 2, 2])
    standardized = ensemble.standardize_by_era(X, eras)
    expected_values_2 = [0.25, 0.75, 0.25, 0.75]
    assert np.allclose(standardized, expected_values_2)

    # Test 3: Different predictions but split across two eras
    X = np.array([0.1, 0.9, 0.9, 0.1])
    eras = np.array([1, 1, 2, 2])
    standardized = ensemble.standardize_by_era(X, eras)
    expected_values_3 = [0.25, 0.75, 0.75, 0.25]
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
    ensemble = NumeraiEnsemble(estimators)
    
    X = np.random.rand(10, 3)
    eras = np.random.randint(1, 5, size=10)
    
    X_fit = np.random.rand(10, 3)
    y_fit = np.random.rand(10)
    ensemble.fit(X_fit, y_fit)

    with pytest.raises(ValueError, match="Predictions for all estimators are constant. No valid predictions to ensemble."):
        ensemble.predict(X, eras)

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
    ensemble = NumeraiEnsemble(estimators)
    
    # Fit the ensemble with dummy data
    X_fit = np.random.rand(10, 3)
    y_fit = np.random.rand(10)
    ensemble.fit(X_fit, y_fit)
    
    X = np.random.rand(10, 3)
    eras = np.random.randint(1, 5, size=10)
    
    # The mock estimator will predict values of 1.5, which are out of [0, 1] range
    with pytest.raises(ValueError, match="Ensembled predictions are not between 0 and 1. Consider checking your estimators."):
        ensemble.predict(X, eras)

def test_donate_weights():
    estimators = [('mock1', ValidMockEstimator()), ('mock2', ValidMockEstimator()),
                  ('mock2', ValidMockEstimator())]
    ensemble = NumeraiEnsemble(estimators, donate_weighted=True)

    # For 3 mock estimators, weights should be [0.25, 0.25, 0.5]
    assert ensemble.weights == [0.25, 0.25, 0.5]

    # For 5 mock estimators, weights should be [0.0625, 0.0625, 0.125, 0.25, 0.5]
    estimators = [('mock1', ValidMockEstimator()), ('mock2', ValidMockEstimator()),
                  ('mock3', ValidMockEstimator()), ('mock4', ValidMockEstimator()),
                  ('mock5', ValidMockEstimator())]
    ensemble = NumeraiEnsemble(estimators, donate_weighted=True)
    assert ensemble.weights == [0.0625, 0.0625, 0.125, 0.25, 0.5]

def test_donate_weights_sum_to_one():
    for n_estimators in range(1, 11):
        estimators = [('mock' + str(i), ValidMockEstimator()) for i in range(n_estimators)]
        ensemble = NumeraiEnsemble(estimators, donate_weighted=True)

        # Assert that the sum of weights is close to 1
        assert np.isclose(sum(ensemble.weights), 1.0)

def test_neutralized_estimator():
    df = create_numerframe("tests/test_assets/train_int8_5_eras.parquet")
    X = df.get_feature_data
    y = df["target"]
    base_estimator = LinearRegression()
    base_estimator.fit(X, y=y)
    vanilla_predictions = base_estimator.predict(X)

    neutralizer = FeatureNeutralizer()
    meta_estimator = NeutralizedEstimator(base_estimator, neutralizer)
    meta_estimator.fit(X, y=y)
    eras = df["era"]
    predictions = meta_estimator.predict(X, eras)
    # Make sure predictions are between 0 and 1
    assert predictions.min() >= 0
    assert predictions.max() <= 1
    # Make sure predictions are different from vanilla predictions
    assert not np.allclose(predictions, vanilla_predictions)

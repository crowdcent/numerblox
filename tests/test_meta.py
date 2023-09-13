

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import rankdata, norm
from sklearn.utils.validation import check_is_fitted

from numerblox.meta import NumeraiEnsemble

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

def test_predict(ensemble, sample_data):
    X, y = sample_data
    ensemble.fit(X, y)
    
    preds = ensemble.predict(X)
    
    # The output of predict should have the same shape as y
    assert preds.shape == y.shape

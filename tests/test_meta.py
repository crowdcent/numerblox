import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from numerblox.meta import MetaEstimator

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
import pytest
import numpy as np
from sklearn.metrics import log_loss
from sklearn.datasets import make_regression, make_classification
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

from numerblox.meta import MetaEstimator, CrossValEstimator
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

from utils import create_classic_sample_data

##### Mock objects #####
@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)

@pytest.fixture
def multiclass_sample_data():
    return make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=3)

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
    with pytest.raises(ValueError):
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


##### CrossValEstimator #####

setup_data = create_classic_sample_data

def dummy_evaluation_func(y_true, y_pred):
    """Example evaluation function."""
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": accuracy}

# Evaluation function that computes the log loss
def multiclass_evaluation_func(y_true, y_pred_proba):
    return {"log_loss": log_loss(y_true, y_pred_proba, labels=[0, 1, 2])}

@pytest.mark.parametrize('cv, estimator', [
    (TimeSeriesSplit(n_splits=2), RandomForestRegressor()),
    (KFold(n_splits=2), RandomForestRegressor()),
])
def test_cross_val_estimator_fit_transform(cv, estimator, setup_data):
    cve = CrossValEstimator(cv=cv, estimator=estimator, evaluation_func=dummy_evaluation_func, verbose=False)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    cve.fit(X, y)
    assert len(cve.estimators_) == cv.get_n_splits(), "Number of fitted estimators should match CV splits."
    assert len(cve.eval_results_) == cv.get_n_splits(), "Evaluation results should match CV splits."
    assert hasattr(cve, "output_shape_"), "Output shape should be set after fitting."
    assert hasattr(cve, "multi_output_"), "Multi-output flag should be set after fitting."
    assert hasattr(cve, "n_outputs_per_model_"), "Number of outputs per model should be set after fitting."
    assert len(cve.eval_results_) > 1
    
    # Transform
    transformed = cve.transform(X)
    expected_num_features = len(cve.estimators_) * cve.n_outputs_per_model_
    assert len(transformed) == len(X), "Transformed shape mismatch."
    assert transformed.shape[1] == expected_num_features, "Transformed shape mismatch."
    
    # Feature names
    feature_names = cve.get_feature_names_out()
    assert len(feature_names) == expected_num_features, "Mismatch in the number of feature names."

def test_invalid_predict_func():
    with pytest.raises(ValueError):
        CrossValEstimator(cv=KFold(n_splits=3), estimator=RandomForestRegressor(), predict_func="invalid_func")

def test_predict_function(setup_data):
    cv = KFold(n_splits=2)
    estimator = RandomForestRegressor()
    cve = CrossValEstimator(cv=cv, estimator=estimator, evaluation_func=dummy_evaluation_func, verbose=False)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    cve.fit(X, y)
    
    transformed = cve.transform(X)
    predicted = cve.predict(X)
    assert np.array_equal(transformed, predicted), "Predict should be the same as transform."

# Multi-output test
def test_cross_val_estimator_multi_output_transform(setup_data):
    cv = KFold(n_splits=2)
    estimator = MultiOutputRegressor(RandomForestRegressor())
    cve = CrossValEstimator(cv=cv, estimator=estimator, evaluation_func=dummy_evaluation_func, verbose=False)
    X, y = setup_data[["feature1", "feature2"]], setup_data[["target", "target_2"]]
    cve.fit(X, y)
    
    # Transform
    transformed = cve.transform(X)
    expected_num_features = len(cve.estimators_) * cve.n_outputs_per_model_
    assert len(transformed) == len(X), "Transformed shape mismatch."
    assert transformed.shape[1] == expected_num_features, "Transformed shape mismatch."

# Test different predict_func values
@pytest.mark.parametrize('predict_func', ['predict', 'predict_proba', 'predict_log_proba'])
def test_different_predict_functions(predict_func, setup_data):
    cv = KFold(n_splits=2)
    # Note: RandomForestRegressor doesn't have 'predict_proba' or 'predict_log_proba'. 
    # So, use a classifier here like RandomForestClassifier
    estimator = RandomForestClassifier()
    cve = CrossValEstimator(cv=cv, estimator=estimator, predict_func=predict_func, verbose=False)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"].round()
    cve.fit(X, y)
    # Transform
    transformed = cve.transform(X)
    expected_num_features = len(cve.estimators_) * cve.n_outputs_per_model_
    assert len(transformed) == len(X), "Transformed shape mismatch."
    assert transformed.shape[1] == expected_num_features, "Transformed shape mismatch."

# Invalid CV strategy
def test_invalid_cv_strategy():
    with pytest.raises(ValueError):
        CrossValEstimator(cv=MockEstimator(), estimator=RandomForestRegressor())
    
# Custom evaluation function behavior
def test_custom_evaluation_func(setup_data):
    def custom_eval(y_true, y_pred):
        return {"custom_metric": np.mean(np.abs(y_true - y_pred))}
    
    cv = KFold(n_splits=3)
    estimator = RandomForestRegressor()
    cve = CrossValEstimator(cv=cv, estimator=estimator, evaluation_func=custom_eval, verbose=False)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    cve.fit(X, y)
    
    for result in cve.eval_results_:
        assert "custom_metric" in result, "Custom metric should be in evaluation results."

# Test for multiclass predict_proba
def test_multiclass_predict_proba(multiclass_sample_data):
    X, y = multiclass_sample_data
    cv = StratifiedKFold(n_splits=3)
    estimator = RandomForestClassifier()
    
    cve = CrossValEstimator(cv=cv, estimator=estimator, predict_func="predict_proba", evaluation_func=multiclass_evaluation_func, verbose=False)
    cve.fit(X, y)
    
    # Verify the shape and values of the transformation
    transformed = cve.transform(X)
    # We expect each model to produce 3 columns of class probabilities (for 3 classes).
    expected_num_features = len(cve.estimators_) * 3  # n_classes
    assert transformed.shape == (len(X), expected_num_features), f"Expected shape {(len(X), expected_num_features)}, but got {transformed.shape}"
    
    # The probabilities should sum up to 1 for each instance
    for i in range(len(X)):
        assert np.isclose(transformed[i, :3].sum(), 1), f"Probabilities do not sum to 1 for instance {i}."
    
    # Evaluation results should contain the log loss
    for result in cve.eval_results_:
        assert "log_loss" in result, f"Log loss not found in evaluation results for fold {cve.eval_results_.index(result)}."

# Test for multiclass predict_log_proba
def test_multiclass_predict_log_proba(multiclass_sample_data):
    X, y = multiclass_sample_data
    cv = StratifiedKFold(n_splits=3)
    estimator = RandomForestClassifier()
    
    cve = CrossValEstimator(cv=cv, estimator=estimator, predict_func="predict_log_proba", evaluation_func=None, verbose=False)
    cve.fit(X, y)
    
    # Verify the shape and values of the transformation
    transformed = cve.transform(X)
    # We expect each model to produce 3 columns of class log-probabilities.
    expected_num_features = len(cve.estimators_) * 3  # n_classes
    assert transformed.shape == (len(X), expected_num_features), f"Expected shape {(len(X), expected_num_features)}, but got {transformed.shape}"

# Test for binary class predict_proba postprocessing
def test_binary_class_postprocess():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    cv = StratifiedKFold(n_splits=3)
    estimator = RandomForestClassifier()
    
    cve = CrossValEstimator(cv=cv, estimator=estimator, predict_func="predict_proba", verbose=False)
    cve.fit(X, y)
    
    # For binary classes, only probabilities of the positive class should be kept
    transformed = cve.transform(X)
    # We predict_proba we get 2 columns per estimator.
    expected_num_features = len(cve.estimators_) * 2
    assert transformed.shape == (len(X), expected_num_features), f"Expected shape {(len(X), expected_num_features)}, but got {transformed.shape}"
    assert (transformed >= 0).all() and (transformed <= 1).all(), "Probabilities should be between 0 and 1."

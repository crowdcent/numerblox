import re

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from utils import create_classic_sample_data

from numerblox.meta import CrossValEstimator, MetaEstimator, MetaPipeline, make_meta_pipeline
from numerblox.neutralizers import FeatureNeutralizer

setup_data = create_classic_sample_data


##### Mock objects #####
@pytest.fixture
def sample_data():
    return make_regression(n_samples=100, n_features=20, noise=0.1)


@pytest.fixture
def multiclass_sample_data():
    return make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=3)


class ConstantMockEstimator(BaseEstimator, RegressorMixin):
    """A mock estimator that always predicts a constant value."""

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

    # Test classifier/regressor
    estimator = LogisticRegression()
    meta_est = MetaEstimator(estimator, predict_func="predict", model_type="classifier")

    assert meta_est.predict_func == "predict"
    assert meta_est.model_type == "classifier"
    assert meta_est.estimator == estimator

    with pytest.raises(ValueError):
        MetaEstimator(estimator, predict_func="invalid_func", model_type="classifier")

    with pytest.raises(AssertionError):
        MetaEstimator(estimator, predict_func="predict", model_type="invalid_type")

    with pytest.raises(AssertionError):
        MetaEstimator(estimator, predict_func="transform", model_type="classifier")


def test_meta_estimator_multioutput():
    # Create dummy multioutput dataset
    X, y = make_regression(n_samples=100, n_features=30, n_targets=3, noise=0.1)

    # Multioutput model
    model_multi = RandomForestRegressor()
    meta_estimator_multi = MetaEstimator(model_multi)
    meta_estimator_multi.fit(X, y)
    assert meta_estimator_multi.multi_output_
    assert meta_estimator_multi.estimator_.__class__ == model_multi.__class__
    transformed = meta_estimator_multi.transform(X)

    assert transformed.shape == (X.shape[0], y.shape[1]), f"Expected shape {(X.shape[0], y.shape[1])}, but got {transformed.shape}"

    # Classification proba
    model_multi = RandomForestClassifier()
    X, y = make_classification(n_samples=100, n_features=30, n_classes=5, n_informative=5)
    meta_est = MetaEstimator(model_multi, predict_func="predict_proba", model_type="classifier")
    meta_est.fit(X, y)
    X_test = np.random.randn(20, 30)
    output = meta_est.transform(X_test)
    assert output.shape == (20, 5), f"Output shape is {output.shape}, but expected (20, 5)"


def test_meta_estimator_get_feature_names_out():
    ensemble = MetaEstimator(ValidMockEstimator(), predict_func="predict")
    ensemble.fit(np.random.uniform(size=(100, 10)), np.random.uniform(size=100))
    assert ensemble.get_feature_names_out() == ["ValidMockEstimator_predict_output"]
    assert ensemble.get_feature_names_out(["a", "b"]) == ["a", "b"]


##### CrossValEstimator #####

setup_data = create_classic_sample_data


def dummy_evaluation_func(y_true, y_pred):
    """Example evaluation function."""
    accuracy = np.mean(y_true == y_pred)
    return {"accuracy": accuracy}


# Evaluation function that computes the log loss
def multiclass_evaluation_func(y_true, y_pred_proba):
    return {"log_loss": log_loss(y_true, y_pred_proba, labels=[0, 1, 2])}


@pytest.mark.parametrize(
    "cv, estimator",
    [
        (TimeSeriesSplit(n_splits=2), RandomForestRegressor()),
        (KFold(n_splits=2), RandomForestRegressor()),
    ],
)
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
@pytest.mark.parametrize("predict_func", ["predict", "predict_proba", "predict_log_proba"])
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
        CrossValEstimator(cv=ConstantMockEstimator(), estimator=RandomForestRegressor())


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


##### MetaPipeline #####
sklearn.set_config(enable_metadata_routing=True)


class MockTransform(BaseEstimator, TransformerMixin):
    """A mock transformer that requires 'era_series' as an argument in its transform method."""

    def __init__(self):
        self.set_predict_request(era_series=True)
        super().__init__()

    def fit(self, X, y=None):
        return self

    def predict(self, X, era_series):
        assert era_series is not None, "era_series should be provided."
        return self.transform(X, era_series)


class MockFinalStep(BaseEstimator, RegressorMixin):
    """A mock final step for the pipeline that requires 'features' and 'era_series' in its predict method."""

    def __init__(self):
        self.set_predict_request(features=True, era_series=True)
        super().__init__()

    def fit(self, X, y=None):
        return self

    def predict(self, X, features, era_series):
        assert features is not None and era_series is not None, "features and era_series should be provided."
        return X


class OneMockEstimator:
    """A mock estimator without extra arguments."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return [1 for _ in range(len(X))]


def test_feature_neutralizer_pipeline(setup_data):
    lr1 = Ridge()
    fn = FeatureNeutralizer(proportion=0.5)
    pipeline = make_meta_pipeline(lr1, fn)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    pipeline.fit(X, y)
    era_series = setup_data["era"]

    result = pipeline.predict(X, features=X, era_series=era_series)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1


def test_meta_pipeline_missing_eras(setup_data):
    # Create a pipeline where a step requires the 'era_series' argument.
    steps = [("mock_transform", MockTransform()), ("final_step", MockFinalStep())]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]

    # Predict without providing 'era_series' should raise a TypeError from MetaEstimator.
    with pytest.raises(TypeError):
        pipeline.fit(X, y).predict(X, features=[])


def test_meta_pipeline_missing_features(setup_data):
    # Create a pipeline with a final step that requires 'features' and 'era_series' arguments.
    final_step = MockFinalStep()
    steps = [("ridge", Ridge()), ("final_step", final_step)]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]
    # Predict without providing 'features' should raise an error.
    with pytest.raises(TypeError, match=re.escape("predict() missing 1 required positional argument: 'features'")):
        pipeline.fit(X, y).predict(X, era_series=[])


def test_meta_pipeline_missing_eras_for_final_step(setup_data):
    # Create a pipeline with a final step that requires 'features' and 'era_series' arguments.
    final_step = MockFinalStep()
    steps = [("ridge", Ridge()), ("final_step", final_step)]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]
    # Predict without providing 'era_series' for the final step should raise an error.
    with pytest.raises(TypeError, match=re.escape("predict() missing 1 required positional argument: 'era_series'")):
        pipeline.fit(X, y).predict(X, features=[])


def test_do_not_wrap_transformer():
    # Define a custom mock transformer with only a transform method (not an estimator)
    class MockOnlyTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    # When passed to the MetaPipeline, it should not be wrapped into a MetaEstimator
    steps = [("mock_only_transform", MockOnlyTransformer())]
    pipeline = MetaPipeline(steps, predict_func="predict")
    assert not isinstance(pipeline.steps[0][1], MetaEstimator), "Transformer was incorrectly wrapped by MetaEstimator!"
    assert isinstance(pipeline.steps[0][1], MockOnlyTransformer), "Transformer class has changed unexpectedly!"


def test_combination_of_transformer_and_estimator():
    # Test that when we have a combination of transformers and estimators, the behavior is as expected
    steps = [
        ("mock_transform", MockTransform()),  # This should be wrapped
        ("mock_only_transform", MockFinalStep()),  # This should not be wrapped as it's the final step
    ]
    pipeline = MetaPipeline(steps, predict_func="predict")

    assert isinstance(pipeline.steps[0][1], MetaEstimator), "Estimator was not wrapped by MetaEstimator!"
    assert isinstance(pipeline.steps[1][1], MockFinalStep), "Final step should remain unchanged!"


def test_meta_pipeline_wrap():
    # Simple pipeline
    pipe = Pipeline([("mock1", MockTransform()), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1], MetaEstimator)  # First step should be wrapped
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)  # Last step should be unchanged


def test_meta_pipeline_nested_pipeline():
    nested_pipe = Pipeline([("mock1", MockTransform()), ("mock2", MockTransform())])

    pipe = Pipeline([("nested", nested_pipe), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1].steps[0][1], MetaEstimator)  # Nested last steps should be wrapped
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)  # Last step should be unchanged


def test_meta_pipeline_feature_union():
    union = FeatureUnion([("mock1", MockTransform()), ("mock2", MockTransform())])

    pipe = Pipeline([("union", union), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1].transformer_list[0][1], MetaEstimator)
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)


def test_meta_pipeline_column_transformer():
    col_trans = ColumnTransformer([("mock1", MockTransform(), [0]), ("mock2", MockTransform(), [1])])

    pipe = Pipeline([("col_trans", col_trans), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1].transformers[0][1], MetaEstimator)
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)


def test_meta_pipeline_deeply_nested():
    deep_nested_pipe = Pipeline([("mock1", MockTransform()), ("union", FeatureUnion([("mock2", MockTransform()), ("mock3", MockTransform())]))])

    pipe = Pipeline([("nested", deep_nested_pipe), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1].steps[1][1].transformer_list[0][1], MetaEstimator)
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)


def test_meta_pipeline_mixed_structures():
    mixed = Pipeline([("union", FeatureUnion([("mock1", MockTransform()), ("col_trans", ColumnTransformer([("mock2", MockTransform(), [0]), ("mock3", MockTransform(), [1])]))])), ("mock4", MockTransform())])

    pipe = Pipeline([("mixed", mixed), ("final", MockFinalStep())])

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1].steps[0][1].transformer_list[1][1].transformers[0][1], MetaEstimator)
    assert isinstance(meta_pipe.steps[1][1], MockFinalStep)


def test_meta_pipeline_estimator_as_transformer():
    pipe = Pipeline(
        [
            ("mock_est", OneMockEstimator()),  # A mock estimator that doesn't have a transform method but might have a predict method
            ("final", MockFinalStep()),
        ]
    )

    meta_pipe = MetaPipeline(pipe.steps)
    assert isinstance(meta_pipe.steps[0][1], MetaEstimator)

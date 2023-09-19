import re
import pytest
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

from numerblox.pipeline import (make_meta_pipeline, MetaPipeline)
from numerblox.neutralizers import FeatureNeutralizer, FeaturePenalizer

from utils import create_classic_sample_data

setup_data = create_classic_sample_data

class MockTransform(BaseEstimator, TransformerMixin):
    """A mock transformer that requires 'eras' as an argument in its transform method."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, eras):
        return X
    
    def predict(self, X, eras):
        return self.transform(X, eras)
    
    def get_params(self, deep=True):
        return {}  # or return actual parameters if any

class MockFinalStep(BaseEstimator, RegressorMixin):
    """A mock final step for the pipeline that requires 'features' and 'eras' in its predict method."""
    def fit(self, X, y=None):
        return self

    def predict(self, X, features, eras):
        return X

def test_feature_neutralizer_pipeline(setup_data):
    lr1 = Ridge()
    fn = FeatureNeutralizer(proportion=0.5)
    pipeline = make_meta_pipeline(lr1, fn)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    pipeline.fit(X, y)
    eras = setup_data["era"]

    result = pipeline.predict(X, features=X, eras=eras)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_meta_pipeline_missing_eras(setup_data):
    # Create a pipeline where a step requires the 'eras' argument.
    steps = [("mock_transform", MockTransform()), ("final_step", MockFinalStep())]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]

    # Predict without providing 'eras' should raise a TypeError from MetaEstimator.
    with pytest.raises(TypeError):
        pipeline.fit(X, y).predict(X, features=[])

def test_meta_pipeline_missing_features(setup_data):
    # Create a pipeline with a final step that requires 'features' and 'eras' arguments.
    steps = [("ridge", Ridge()), ("final_step", MockFinalStep())]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]
    # Predict without providing 'features' should raise an error.
    with pytest.raises(TypeError, match=re.escape("predict() missing 1 required positional argument: 'features'")):
        pipeline.fit(X, y).predict(X, eras=[])

def test_meta_pipeline_missing_eras_for_final_step(setup_data):
    # Create a pipeline with a final step that requires 'features' and 'eras' arguments.
    steps = [("ridge", Ridge()), ("final_step", MockFinalStep())]
    pipeline = MetaPipeline(steps)

    X = setup_data[["feature1", "feature2"]]
    y = setup_data["target"]
    # Predict without providing 'eras' for the final step should raise an error.
    with pytest.raises(TypeError, match=re.escape("predict() missing 1 required positional argument: 'eras'")):
        pipeline.fit(X, y).predict(X, features=[])

# TODO Test that these objects work well within other Pipelines, FeatureUnions, and ColumnTransformers.

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
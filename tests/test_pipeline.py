import numpy as np 
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from numerblox.pipeline import make_numerai_pipeline, make_numerai_union
from numerblox.neutralizers import FeatureNeutralizer, FeaturePenalizer

from utils import create_classic_sample_data

setup_data = create_classic_sample_data

def test_feature_neutralizer_pipeline(setup_data):
    lr1 = Ridge()
    fn = FeatureNeutralizer(proportion=0.5)
    pipeline = make_numerai_pipeline(lr1, fn)
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    pipeline.fit(X, y)
    eras = setup_data["era"]

    result = pipeline.predict(X, features=X, eras=eras)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    assert result.min() >= 0
    assert result.max() <= 1

def test_feature_neutralizer_featureunion(setup_data):
    onehot = OneHotEncoder(sparse_output=False)
    pipeline = make_numerai_pipeline(Ridge(), FeatureNeutralizer(proportion=0.5))
    union = make_numerai_union(onehot, pipeline)

    # Fitting the union
    X, y = setup_data[["feature1", "feature2"]], setup_data["target"]
    union.fit(X, y)

    eras = setup_data["era"]

    # Making predictions
    result = union.transform(X, numeraipipeline={"features": X, "eras": eras})

    # Your assertions
    assert isinstance(result, np.ndarray)
    assert len(result) == len(setup_data)
    # All one hot should be in the right place with right values.
    assert np.all(np.isin(result[:, :-1], [0, 1]))
    # All pipeline predictions should be between 0 and 1.
    assert np.all((0 <= result[:, -1]) & (result[:, -1] <= 1))


# TODO Test that these object work well within other Pipelines, FeatureUnions, and ColumnTransformers.

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
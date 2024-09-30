import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from numerblox.prediction_loaders import ExamplePredictions


def test_example_predictions_basic():
    ep = ExamplePredictions()
    preds = ep.fit_transform(None)
    # Check all values are between 0 and 1
    assert preds["prediction"].min() >= 0
    assert preds["prediction"].max() <= 1
    assert isinstance(preds, pd.DataFrame)
    assert issubclass(ExamplePredictions, (BaseEstimator, TransformerMixin))


def test_example_predictions_pipeline():
    # Create dummy dataset
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
    X = pd.DataFrame(X)

    # Create pipeline with standard scaler and example predictions
    pipeline = Pipeline([("scaler", StandardScaler()), ("predictions", ExamplePredictions())])
    # Get results
    preds = pipeline.fit_transform(X, y)

    # Check all values are between 0 and 1
    assert preds["prediction"].min() >= 0
    assert preds["prediction"].max() <= 1
    assert isinstance(preds, pd.DataFrame)


def test_example_predictions_feature_union():
    # Get predictions in basic setting to compare output
    ep = ExamplePredictions()
    preds = ep.fit_transform(None)

    # Dummy data
    X, _ = make_regression(n_samples=len(preds), n_features=2, noise=0.1)

    # Create feature union
    combined_features = FeatureUnion([("standard", StandardScaler()), ("example", ExamplePredictions())])

    # Transform data
    X_transformed = combined_features.fit_transform(X)

    # Ensure the transformation worked
    assert np.allclose(X_transformed[:, -1], preds["prediction"].values)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == 3


def test_example_predictions_get_feature_names_out():
    ep = ExamplePredictions()
    assert ep.get_feature_names_out() == ["v5.0/live_example_preds"]
    assert ep.get_feature_names_out(["a", "b"]) == ["a", "b"]


def test_example_predictions_keep_files():
    # Test with keep_files = True
    ep_keep = ExamplePredictions(keep_files=True)
    ep_keep.fit_transform(None)
    assert os.path.isdir(ep_keep.downloader.dir), "Directory should be kept with keep_files=True"
    assert os.path.exists(ep_keep.dest_path), "File should be kept with keep_files=True"
    # Clean up
    ep_keep.downloader.remove_base_directory()

    # Test with keep_files = False
    ep_remove = ExamplePredictions(keep_files=False)
    ep_remove.fit_transform(None)
    assert not os.path.isdir(ep_remove.downloader.dir), "Directory should be removed with keep_files=False"

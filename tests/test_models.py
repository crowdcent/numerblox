import numpy as np
from utils import create_classic_sample_data

from numerblox.models import EraBoostedXGBRegressor

setup_data = create_classic_sample_data


def test_initialization():
    model = EraBoostedXGBRegressor()
    assert model.proportion == 0.5
    assert model.trees_per_step == 10
    assert model.num_iters == 200
    assert model.n_estimators == 100

    custom_model = EraBoostedXGBRegressor(proportion=0.3, trees_per_step=5, num_iters=10)
    assert custom_model.proportion == 0.3
    assert custom_model.trees_per_step == 5
    assert custom_model.num_iters == 10


def test_fit_method(setup_data):
    model = EraBoostedXGBRegressor(proportion=0.5, num_iters=5, n_estimators=100, max_depth=3, learning_rate=0.1)
    X, y, era_series = setup_data[["feature1", "feature2"]], setup_data["target"], setup_data["era"]
    initial_tree_count = model.n_estimators

    model.fit(X, y, era_series=era_series, verbose=500)

    assert model.n_estimators > initial_tree_count
    # Check if the final number of trees is as expected
    expected_final_tree_count = initial_tree_count + (model.num_iters - 1) * model.trees_per_step
    assert model.n_estimators == expected_final_tree_count


def test_predictions(setup_data):
    model = EraBoostedXGBRegressor(num_iters=5, proportion=0.5, n_estimators=100, learning_rate=0.1, max_depth=3)
    X, y, era_series = setup_data[["feature1", "feature2"]], setup_data["target"], setup_data["era"]
    model.fit(X, y, era_series=era_series)

    predictions = model.predict(X)
    assert len(predictions) == len(X)
    # Check that predictions are not constant.
    assert len(set(predictions)) > 1
    # Check that it has fitted the data reasonably well.
    correlation = np.corrcoef(predictions, y)[0, 1]
    assert correlation > 0.8

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from numerblox.model_upload import EnsembleModel  # Use the imported EnsembleModel

import pytest


def create_synthetic_data():
    num_eras = 50
    rows_per_era = 4000
    num_features = 310

    # Generate feature column names
    feature_cols = [f'feature_{i}' for i in range(num_features)]

    # Generate synthetic data
    data = []
    for era in range(1, num_eras + 1):
        era_data = pd.DataFrame(np.random.randn(rows_per_era, num_features), columns=feature_cols)
        era_data['era'] = f'era{era}'
        era_data['target'] = np.random.uniform(0, 1, rows_per_era)
        data.append(era_data)

    # Concatenate all eras into a single DataFrame
    df = pd.concat(data, ignore_index=True)

    return df, feature_cols


def train_lightgbm_models(X, y):
    dtrain = lgb.Dataset(X, label=y)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model_1 = lgb.train(params, dtrain, num_boost_round=100)
    model_2 = lgb.train(params, dtrain, num_boost_round=100)
    return model_1, model_2


def test_two_layer_ensemble():
    # Create synthetic data
    df, feature_cols = create_synthetic_data()
    X = df[feature_cols]
    y = df['target']

    # Train two LightGBM models for the first layer
    model_1, model_2 = train_lightgbm_models(X, y)

    # Prepare live features data for prediction
    live_features = pd.DataFrame(np.random.rand(4000, len(feature_cols)), columns=feature_cols)
    live_features['id'] = range(4000)
    live_features.set_index('id', inplace=True)

    # Generate predictions for the first layer
    pred_1 = model_1.predict(live_features[feature_cols])
    pred_2 = model_2.predict(live_features[feature_cols])

    # Combine first layer predictions into a DataFrame
    first_layer_preds = pd.DataFrame({'pred_1': pred_1, 'pred_2': pred_2}, index=live_features.index)

    # Train a second layer model (meta model) on the outputs of the first layer models
    meta_model = LinearRegression()
    meta_model.fit(first_layer_preds, y[:4000])  # Use the target from the original data for simplicity

    # Define ensemble specification for a 2-layer ensemble
    ensemble_spec = {
        1: {
            'models': [model_1, model_2],
            'weights': [0.5, 0.5]
        },
        2: {
            'models': [meta_model],
            'weights': [1.0]
        }
    }

    # Create EnsembleModel instance using the existing EnsembleModel class
    ensemble_model = EnsembleModel(ensemble_spec)

    # Generate predictions using the EnsembleModel
    predictions_two_layer_ensemble = ensemble_model.predict(live_features)

    # Manually compute the predictions for the 2-layer ensemble
    # Manually compute first layer average predictions
    manual_second_layer_input = pd.DataFrame({
        'pred_1': first_layer_preds['pred_1'],
        'pred_2': first_layer_preds['pred_2']
    }, index=live_features.index)

    # Predict using the second layer model
    manual_final_predictions = meta_model.predict(manual_second_layer_input)
    manual_final_ranked_predictions = pd.Series(manual_final_predictions, index=live_features.index).rank(pct=True, method='first')

    # Assert that the predictions are approximately equal
    assert np.allclose(predictions_two_layer_ensemble, manual_final_ranked_predictions, atol=1e-8), \
        "Predictions from the 2-layer ensemble model should be approximately equal."


if __name__ == "__main__":
    pytest.main()

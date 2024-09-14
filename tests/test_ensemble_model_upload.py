import numpy as np
import pandas as pd
import lightgbm as lgb
from numerblox.model_upload import EnsembleModel


def create_synthetic_data():
    num_eras = 20
    rows_per_era = 300
    num_features = 50

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


def train_model(X, y):
    dtrain = lgb.Dataset(X, label=y)
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model = lgb.train(params, dtrain, num_boost_round=100)
    return model


def test_ensemble_predictions():
    # Create synthetic data
    df, feature_cols = create_synthetic_data()
    X = df[feature_cols]
    y = df['target']

    # Train a single LightGBM model
    model = train_model(X, y)

    # Prepare live features data for prediction
    live_features = pd.DataFrame(np.random.rand(4000, len(feature_cols)), columns=feature_cols)
    live_features['id'] = range(4000)
    live_features.set_index('id', inplace=True)

    # Predict with single model
    single_predictions = model.predict(live_features[feature_cols])
    ranked_single_predictions = pd.Series(single_predictions, index=live_features.index).rank(pct=True, method='first')

    # Create ensemble specification using the same model twice with weights [0.5, 0.5]
    ensemble_spec = {
        1: {
            'models': [model, model],
            'weights': [0.5, 0.5]
        }
    }

    # Create EnsembleModel instance
    ensemble_model = EnsembleModel(ensemble_spec)

    # Predict with ensemble model
    ensemble_predictions = ensemble_model.predict(live_features)
    ranked_ensemble_predictions = pd.Series(ensemble_predictions, index=live_features.index)

    # Assert that the predictions are approximately equal
    assert np.allclose(ranked_single_predictions, ranked_ensemble_predictions, atol=1e-6), \
        "Predictions from single model and ensemble model should be approximately equal."


if __name__ == "__main__":
    test_ensemble_predictions()

import pytest
import numpy as np
import pandas as pd

from numerblox.numerframe import NumerFrame

# Fixture to create a dummy dataframe
@pytest.fixture
def dummy_dataframe():
    df = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6],
        'target': [7, 8, 9],
        'target2': [10, 11, 12],
        'era': ['001', '002', '002'],
        'prediction': np.random.uniform(size=3),
    })
    return df

@pytest.fixture
def create_classic_sample_data():
    data = {
        "feature1": [1, 2, 3, 4, 3, 2, 1],
        "feature2": [4, 3, 2, 1, 3, 1, 2],
        "prediction": [0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4],
        "target": [0, 1, 0, 1, 0.25, 0.75, 0.5],
        "target_2": [0, 0.25, 0.75, 0.50, 0.25, 0.75, 0.5],
        "era": ["era1", "era2", "era1", "era2", "era1", "era2", "era1"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def create_signals_sample_data():
    instances = []
    tickers = ["ABC.US", "DEF.US", "GHI.US"]
    for ticker in tickers:
        price = np.random.randint(10, 100)
        for i in range(100):
            price += np.random.uniform(-1, 1)
            instances.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "open": price - 0.05,
                    "high": price + 0.02,
                    "low": price - 0.01,
                    "close": price,
                    "volume": np.random.randint(1000, 10000),
                    "target": np.random.uniform(),
                    "target_2": np.random.uniform(),
                    "prediction": np.random.uniform(),
                    "prediction_random": np.random.uniform(),
                }
            )
    dummy_df = NumerFrame(instances)
    return dummy_df
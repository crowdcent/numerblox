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
        'era': ['001', '002', '002'],
        'prediction': np.random.uniform(size=3),
    })
    return df

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
                    "prediction": np.random.uniform(),
                    "prediction_random": np.random.uniform(),
                }
            )
    dummy_df = NumerFrame(instances)
    return dummy_df
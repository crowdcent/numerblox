import os
import pytest
import numpy as np
import pandas as pd
from uuid import uuid4
from random import choices
from copy import deepcopy
from datetime import datetime
from string import ascii_uppercase
from dateutil.relativedelta import relativedelta, FR

from numerblox.submission import (NumeraiClassicSubmitter, NumeraiSignalsSubmitter, Key)


TARGET_NAME = "prediction"

def _create_random_classic_df():
    # Create random dataframe
    n_rows = 100
    test_dataf = pd.DataFrame(np.random.uniform(size=n_rows), columns=[TARGET_NAME])
    test_dataf["id"] = [uuid4() for _ in range(n_rows)]
    test_dataf = test_dataf.set_index("id")
    return test_dataf

def create_random_signals_df(n_rows=1000):
    signals_test_dataf = pd.DataFrame(
        np.random.uniform(size=(n_rows, 1)), columns=["signal"]
    )
    signals_test_dataf["ticker"] = [
        "".join(choices(ascii_uppercase, k=4)) for _ in range(n_rows)
    ]
    last_friday = str((datetime.now() + relativedelta(weekday=FR(-1))).date()).replace("-", "")
    signals_test_dataf['last_friday'] = last_friday
    signals_test_dataf['data_type'] = 'live'
    return signals_test_dataf

def test_classic_submitter():
    # Initialization
    test_dir = "test_sub"
    classic_key = Key(pub_id="Hello", secret_key="World")
    num_sub = NumeraiClassicSubmitter(directory_path=test_dir, key=classic_key)
    assert num_sub.dir.is_dir()

    # Save CSV
    test_dataf = _create_random_classic_df()
    file_name = "test.csv"
    num_sub.save_csv(dataf=test_dataf, file_name=file_name, cols=TARGET_NAME)
    num_sub.save_csv(dataf=test_dataf, file_name="test2.csv", cols=TARGET_NAME)
    assert (num_sub.dir / file_name).is_file()

    # Combine CSVs
    combined = num_sub.combine_csvs(["test_sub/test.csv", "test_sub/test2.csv"], aux_cols=['id'])
    assert combined.columns == [TARGET_NAME]

    # Test that saving breaks if range is invalid.
    with pytest.raises(ValueError):
        invalid_signal = deepcopy(test_dataf)
        invalid_signal.iloc[0][TARGET_NAME] += 10
        num_sub.save_csv(
            invalid_signal,
            file_name="should_not_save.csv",
            cols=TARGET_NAME,
        )
    
    # Wind down
    num_sub.remove_base_directory()
    assert not os.path.exists(test_dir)

def test_signals_submitter():
    # Initialization
    test_dir = "test_sub"
    signals_key = Key(pub_id="Hello", secret_key="World")
    signals_sub = NumeraiSignalsSubmitter(directory_path=test_dir, key=signals_key)
    assert signals_sub.dir.is_dir()

    # Save CSVs
    test_dataf = create_random_signals_df()
    signals_cols = ["signal", "ticker", "data_type", "last_friday"]
    file_name = "signals_test.csv"
    signals_sub.save_csv(dataf=test_dataf, file_name=file_name, cols=signals_cols)
    signals_sub.save_csv(dataf=test_dataf, file_name="signals_test2.csv", cols=signals_cols)

    combined_signals = signals_sub.combine_csvs(csv_paths=[
        "test_sub/signals_test.csv", 
        "test_sub/signals_test2.csv"
        ],
        aux_cols=['ticker', 'last_friday', 'data_type'],
        era_col='last_friday',
        pred_col='signal'
        )
    assert combined_signals.columns == ['signal']

    # Test that saving breaks if range is invalid.
    with pytest.raises(ValueError):
        invalid_signal = deepcopy(test_dataf)
        invalid_signal.loc[0, "signal"] += 10
        signals_sub.save_csv(
            invalid_signal,
            file_name="should_not_save.csv",
            cols=list(invalid_signal.columns),
        )

    # Test that saving breaks if ticker is invalid.
    with pytest.raises(NotImplementedError):
        invalid_ticker = deepcopy(test_dataf)
        invalid_ticker = invalid_ticker.rename(
            {"ticker": "not_a_valid_ticker_format"}, axis=1
        )
        signals_sub.save_csv(
            invalid_ticker,
            file_name="should_not_save.csv",
            cols=list(invalid_ticker.columns),
        )
    # Wind down
    signals_sub.remove_base_directory()
    assert not os.path.exists(test_dir)

# Tests for NumerBaySubmitter
def test_numerbay_submitter():
    pass
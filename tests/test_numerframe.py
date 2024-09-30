import pytest
import numpy as np
import pandas as pd
from numerai_era_data.date_utils import ERA_ONE_START

from numerblox.numerframe import NumerFrame, create_numerframe
from numerblox.feature_groups import FNCV3_FEATURES, SMALL_FEATURES, MEDIUM_FEATURES, V5_FEATURE_GROUP_MAPPING

TEST_FILE_PATH = "tests/test_assets/val_3_eras.parquet"
dataset = pd.read_parquet(TEST_FILE_PATH)

def test_numerframe_initialization():
    nf = NumerFrame(dataset)
    assert isinstance(nf, NumerFrame)
    assert nf.meta == {'era_col': 'era'}
    assert nf.meta.era_col == 'era'

def test_get_feature_data():
    nf = NumerFrame(dataset)
    features = nf.get_feature_data
    assert isinstance(features, NumerFrame)
    assert all([col.startswith("feature_") for col in features.columns.tolist()])

def test_get_pattern_data():
    nf = NumerFrame(dataset)
    xerxes_targets = nf.get_pattern_data('xerxes')
    assert isinstance(xerxes_targets, NumerFrame)
    assert xerxes_targets.columns.tolist() == ['target_xerxes_20', 'target_xerxes_60']

def test_get_target_data():
    nf = NumerFrame(dataset)
    targets = nf.get_target_data
    assert isinstance(targets, NumerFrame)
    assert all([col.startswith("target") for col in targets.columns.tolist()])

def test_get_single_target_data():
    nf = NumerFrame(dataset)
    single_target = nf.get_single_target_data
    assert isinstance(single_target, NumerFrame)
    assert single_target.columns.tolist() == ['target']

def test_get_prediction_data():
    nf = NumerFrame(dataset)
    preds = nf.get_prediction_data
    assert isinstance(preds, NumerFrame)
    assert preds.columns.tolist() == []

def test_get_column_selection():
    nf = NumerFrame(dataset)
    result = nf.get_column_selection(['feature_itinerant_hexahedral_photoengraver'])
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == ['feature_itinerant_hexahedral_photoengraver']

def test_get_aux_data():
    nf = NumerFrame(dataset)
    aux_data = nf.get_aux_data
    assert isinstance(aux_data, NumerFrame)
    assert aux_data.columns.tolist() == ['era', 'data_type']

def test_get_era_data():
    nf = NumerFrame(dataset)
    era_data = nf.get_era_data
    assert isinstance(era_data, NumerFrame)
    assert era_data.columns.tolist() == ['era']

def test_get_prediction_aux_data():
    nf = NumerFrame(dataset)
    nf['prediction'] = 1
    nf = NumerFrame(nf)
    pred_aux = nf.get_prediction_aux_data
    assert isinstance(pred_aux, NumerFrame)
    assert pred_aux.columns.tolist() == ['prediction', 'era', 'data_type']

def test_get_feature_target_pair():
    nf = NumerFrame(dataset)
    X, y = nf.get_feature_target_pair()
    assert isinstance(X, NumerFrame)
    assert X.columns.tolist() == nf.get_feature_data.columns.tolist()
    assert y.columns.tolist() == ['target']

def test_get_feature_target_pair_multi_target():
    nf = NumerFrame(dataset)
    X, y = nf.get_feature_target_pair(multi_target=True)
    assert isinstance(X, NumerFrame)
    assert X.columns.tolist() == nf.get_feature_data.columns.tolist()
    assert y.columns.tolist() == nf.get_target_data.columns.tolist()

def test_get_fncv3_features():
    nf = NumerFrame(dataset)
    result = nf.get_fncv3_feature_data
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == FNCV3_FEATURES

def test_get_small_features():
    nf = NumerFrame(dataset)
    result = nf.get_small_feature_data
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == SMALL_FEATURES

def test_get_medium_features():
    nf = NumerFrame(dataset)
    result = nf.get_medium_feature_data
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == MEDIUM_FEATURES

def test_get_unique_eras():
    nf = NumerFrame(dataset)
    result = nf.get_unique_eras
    assert isinstance(result, list)
    assert result == ["0575", "0576", "0577"]

def test_get_feature_group():
    # Test with a valid group name
    nf = NumerFrame(dataset)
    result = nf.get_feature_group("rain")
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == V5_FEATURE_GROUP_MAPPING["rain"]

    # Test with an invalid group name
    with pytest.raises(AssertionError, match=r".*not found in.*"):
        nf.get_feature_group("group_invalid")

def test_get_last_n_eras():
    nf = NumerFrame(dataset)
    result = nf.get_last_n_eras(2)
    assert isinstance(result, NumerFrame)
    assert result[nf.meta.era_col].unique().tolist() == ["0576", "0577"]
    assert result.shape == (11313, 2415)

def test_get_era_batch():
    nf = NumerFrame(dataset)
    eras = ["0575", "0576"]
    X, y = nf.get_era_batch(eras=eras)
    assert isinstance(X, np.ndarray)
    assert X.shape == (11230, 2376)
    assert y.shape == (11230, 37)

def test_get_era_from_date():
    nf = NumerFrame(dataset)
    era = nf.get_era_from_date(pd.Timestamp('2016-01-01'))
    assert isinstance(era, int)
    assert era == 677

    era1 = nf.get_era_from_date(pd.Timestamp(ERA_ONE_START))
    assert isinstance(era1, int) 
    assert era1 == 1

def test_get_date_from_era():
    nf = NumerFrame(dataset)
    date = nf.get_date_from_era(era=4)
    assert isinstance(date, pd.Timestamp)
    assert date == pd.Timestamp('2003-02-01')

    date1 = nf.get_date_from_era(era=1)
    assert isinstance(date1, pd.Timestamp)
    assert date1 == pd.Timestamp(ERA_ONE_START)

def test_get_dates_from_era_col():
    nf = NumerFrame(dataset).iloc[:5]
    result = nf.get_dates_from_era_col
    assert isinstance(result, pd.Series)
    assert all(result.index == nf.index[:5])
    assert result.tolist() == [pd.Timestamp('2014-01-11 00:00:00')] * len(result)

def test_get_eras_from_date_col():
    dataset_copy = dataset.copy()
    # Use a smaller range of dates
    dataset_copy['date'] = [pd.Timestamp(ERA_ONE_START) + pd.Timedelta(days=i) for i in range(len(dataset_copy))]
    dataset_copy = dataset_copy.drop(columns="era")
    nf = NumerFrame(dataset_copy.iloc[:5])
    result = nf.get_eras_from_date_col
    assert isinstance(result, pd.Series)
    assert all(result.index == nf.index[:5])
    assert result.tolist() == [1, 1, 1, 1, 1]

def test_get_era_range():
    nf = NumerFrame(dataset)
    result = nf.get_era_range(start_era=575, end_era=576)
    assert isinstance(result, NumerFrame)
    assert result[nf.meta.era_col].unique().tolist() == ["0575", "0576"]
    assert result.shape == (11230, 2415)

    with pytest.raises(AssertionError):
        no_era_dataset = dataset.drop("era", axis="columns")
        no_era_dataset["date"] = pd.Timestamp('2016-01-01')
        nf = NumerFrame(no_era_dataset)
        nf.get_era_range(start_era=1, end_era=3)
    # Negative era
    with pytest.raises(AssertionError):
        nf.get_era_range(-1, 5)
    # End era before start era
    with pytest.raises(AssertionError):
        nf.get_era_range(20, 3)
    # Start era not int
    with pytest.raises(AssertionError):
        nf.get_era_range("0001", 2)
    # End era not int
    with pytest.raises(AssertionError):
        nf.get_era_range(1, "0002")

def test_get_date_range():
    date_col_dataset = dataset.drop("era", axis="columns")
    date_col_dataset["date"] = [pd.Timestamp('2016-01-01') + pd.Timedelta(days=i) for i in range(0, len(date_col_dataset))]
    nf = NumerFrame(date_col_dataset)
    result = nf.get_date_range(start_date=pd.Timestamp('2016-01-01'), end_date=pd.Timestamp('2016-01-03'))
    assert isinstance(result, NumerFrame)
    assert result[nf.meta.era_col].unique().tolist() == [pd.Timestamp('2016-01-01'), pd.Timestamp('2016-01-02'), 
                                                         pd.Timestamp('2016-01-03')]
    assert result.shape == (3, 2415)

    # End date before start date
    with pytest.raises(AssertionError):
        nf.get_date_range(pd.Timestamp('2022-01-05'), pd.Timestamp('2022-01-01'))
    # Date before era 1
    with pytest.raises(AssertionError):
        nf.get_date_range(pd.Timestamp('1970-01-05'), pd.Timestamp('1971-01-10'))
    # Start date not pd.Timestamp
    with pytest.raises(AssertionError):
        nf.get_date_range("2016-01-01", pd.Timestamp('2016-01-10'))
    # End date not pd.Timestamp
    with pytest.raises(AssertionError):
        nf.get_date_range(pd.Timestamp('2016-01-01'), "2016-01-10")

def test_create_numerframe():
    nf = create_numerframe(TEST_FILE_PATH)
    assert isinstance(nf, NumerFrame)

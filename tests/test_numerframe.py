import pytest
import numpy as np
import pandas as pd

from numerblox.numerframe import NumerFrame, create_numerframe
from numerblox.feature_groups import (V4_2_FEATURE_GROUP_MAPPING, FNCV3_FEATURES, 
                                      SMALL_FEATURES, MEDIUM_FEATURES, V2_EQUIVALENT_FEATURES, 
                                      V3_EQUIVALENT_FEATURES)

dataset = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")

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
    jerome_targets = nf.get_pattern_data('jerome')
    assert isinstance(jerome_targets, NumerFrame)
    assert jerome_targets.columns.tolist() == ['target_jerome_v4_20', 'target_jerome_v4_60']

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
    result = nf.get_column_selection(['target_jerome_v4_20'])
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == ['target_jerome_v4_20']

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

def test_get_v2_equivalent_features():
    nf = NumerFrame(dataset)
    result = nf.get_v2_equivalent_feature_data
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == V2_EQUIVALENT_FEATURES

def test_get_v3_equivalent_features():
    nf = NumerFrame(dataset)
    result = nf.get_v3_equivalent_feature_data
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == V3_EQUIVALENT_FEATURES

def test_get_unique_eras():
    nf = NumerFrame(dataset)
    result = nf.get_unique_eras
    assert isinstance(result, list)
    assert result == ['0001', '0002', '0003', '0004', '0005']

def test_get_feature_group():
    # Test with a valid group name
    nf = NumerFrame(dataset)
    result = nf.get_feature_group("rain")
    assert isinstance(result, NumerFrame)
    assert result.columns.tolist() == V4_2_FEATURE_GROUP_MAPPING["rain"]

    # Test with an invalid group name
    with pytest.raises(AssertionError, match=r".*not found in.*"):
        nf.get_feature_group("group_invalid")

def test_get_last_n_eras():
    nf = NumerFrame(dataset)
    result = nf.get_last_n_eras(2)
    assert isinstance(result, NumerFrame)
    assert result[nf.meta.era_col].unique().tolist() == ['0004', '0005']
    assert result.shape == (4805, 2183)

def test_get_era_batch():
    nf = NumerFrame(dataset)
    eras = ['0004', '0005']
    X, y = nf.get_era_batch(eras=eras, convert_to_tf=False)
    assert isinstance(X, np.ndarray)
    assert X.shape == (4805, 2132)
    assert y.shape == (4805, 49)

def test_create_numerframe():
    file_path = "tests/test_assets/train_int8_5_eras.parquet"
    nf = create_numerframe(file_path)
    assert isinstance(nf, NumerFrame)

import pytest
import pandas as pd

from numerblox.numerframe import NumerFrame, create_numerframe
from numerblox.feature_groups import V4_2_FEATURE_GROUP_MAPPING, FNCV3_FEATURES

dataset = pd.read_parquet("tests/test_assets/train_int8_5_eras.parquet")

def test_numerframe_initialization():
    nf = NumerFrame(dataset)
    assert isinstance(nf, NumerFrame)
    assert nf.meta == {'era_col': 'era'}
    assert nf.meta.era_col == 'era'

def test_get_feature_data():
    nf = NumerFrame(dataset)
    assert all([col.startswith("feature_") for col in nf.get_feature_data.columns.tolist()])

def test_get_pattern_data():
    nf = NumerFrame(dataset)
    assert nf.get_pattern_data('jerome').columns.tolist() == ['target_jerome_v4_20', 'target_jerome_v4_60']

def test_get_target_data():
    nf = NumerFrame(dataset)
    assert all([col.startswith("target") for col in nf.get_target_data.columns.tolist()])

def test_get_single_target_data():
    nf = NumerFrame(dataset)
    assert nf.get_single_target_data.columns.tolist() == ['target']

def test_get_prediction_data():
    nf = NumerFrame(dataset)
    assert nf.get_prediction_data.columns.tolist() == []

def test_get_column_selection():
    nf = NumerFrame(dataset)
    assert nf.get_column_selection(['target_jerome_v4_20']).columns.tolist() == ['target_jerome_v4_20']

def test_get_aux_data():
    nf = NumerFrame(dataset)
    assert nf.get_aux_data.columns.tolist() == ['era', 'data_type']

def test_get_prediction_aux_data():
    nf = NumerFrame(dataset)
    nf['prediction'] = 1
    nf = NumerFrame(nf)
    assert nf.get_prediction_aux_data.columns.tolist() == ['prediction', 'era', 'data_type']

def test_get_feature_target_pair():
    nf = NumerFrame(dataset)
    X, y = nf.get_feature_target_pair()
    assert X.columns.tolist() == nf.get_feature_data.columns.tolist()
    assert y.columns.tolist() == ['target']

def test_get_feature_target_pair_multi_target():
    nf = NumerFrame(dataset)
    X, y = nf.get_feature_target_pair(multi_target=True)
    assert X.columns.tolist() == nf.get_feature_data.columns.tolist()
    assert y.columns.tolist() == nf.get_target_data.columns.tolist()

def test_get_fncv3_features():
    nf = NumerFrame(dataset)
    result = nf.get_fncv3_features
    assert result.columns.tolist() == FNCV3_FEATURES


def test_get_feature_group():
    # Test with a valid group name
    nf = NumerFrame(dataset)
    result = nf.get_feature_group("rain")
    assert result.columns.tolist() == V4_2_FEATURE_GROUP_MAPPING["rain"]

    # Test with an invalid group name
    with pytest.raises(AssertionError, match=r".*not found in.*"):
        nf.get_feature_group("group_invalid")

def test_get_era_batch():
    nf = NumerFrame(dataset)
    eras = ['0004', '0005']
    X, y = nf.get_era_batch(eras=eras, convert_to_tf=False)
    assert X.shape == (4805, 2132)
    assert y.shape == (4805, 49)

def test_create_numerframe():
    # Here you would use the provided "test_assets/train_int8_5_eras.parquet" file
    file_path = "tests/test_assets/train_int8_5_eras.parquet"
    nf = create_numerframe(file_path)
    assert isinstance(nf, NumerFrame)

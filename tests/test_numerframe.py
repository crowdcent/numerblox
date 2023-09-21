from numerblox.numerframe import NumerFrame, create_numerframe

from utils import dummy_dataframe

def test_numerframe_initialization(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert isinstance(nf, NumerFrame)
    assert nf.meta == {'era_col': 'era'}
    assert nf.meta.era_col == 'era'

def test_get_feature_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_feature_data.columns.tolist() == ['feature_1', 'feature_2']

def test_get_pattern_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_pattern_data('_1').columns.tolist() == ['feature_1']

def test_get_target_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_target_data.columns.tolist() == ['target', 'target2']

def test_get_single_target_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_single_target_data.columns.tolist() == ['target']

def test_get_prediction_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_prediction_data.columns.tolist() == ['prediction']

def test_get_column_selection(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_column_selection(['feature_1', 'target']).columns.tolist() == ['feature_1', 'target']

def test_get_aux_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_aux_data.columns.tolist() == ['era']

def test_get_prediction_aux_data(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    assert nf.get_prediction_aux_data.columns.tolist() == ['prediction', 'era']

def test_get_feature_target_pair(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    X, y = nf.get_feature_target_pair()
    assert X.columns.tolist() == ['feature_1', 'feature_2']
    assert y.columns.tolist() == ['target']

def test_get_feature_target_pair_multi_target(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    X, y = nf.get_feature_target_pair(multi_target=True)
    assert X.columns.tolist() == ['feature_1', 'feature_2']
    assert y.columns.tolist() == ['target', 'target2']

# Test get_era_batch functionality
def test_get_era_batch(dummy_dataframe):
    nf = NumerFrame(dummy_dataframe)
    eras = ['001', '002']
    X, y = nf.get_era_batch(eras=eras, convert_to_tf=False)
    assert X.shape == (3, 2)
    assert y.shape == (3, 2)

def test_create_numerframe():
    # Here you would use the provided "test_assets/train_int8_5_eras.parquet" file
    file_path = "tests/test_assets/train_int8_5_eras.parquet"
    nf = create_numerframe(file_path)
    assert isinstance(nf, NumerFrame)
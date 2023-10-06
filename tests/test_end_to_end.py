import pytest
from xgboost import XGBRegressor
from sklego.preprocessing import ColumnSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_union, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from numerblox.numerframe import create_numerframe
from numerblox.preprocessing import GroupStatsPreProcessor
from numerblox.meta import CrossValEstimator, make_meta_pipeline, MetaEstimator
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.neutralizers import FeatureNeutralizer

@pytest.fixture(scope="module")
def setup_data():
    df = create_numerframe("tests/test_assets/train_int8_5_eras.parquet")
    return df

def test_neutralized_xgboost_pipeline(setup_data):
    df = setup_data

    X, y = df.get_feature_target_pair(multi_target=False)
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()
    eras = df.get_era_data
    features = df.get_feature_data

    # Preprocessing
    gpp = GroupStatsPreProcessor(groups=['sunshine', 'rain'])
    fncv3_selector = ColumnSelector(fncv3_cols)
    preproc_pipe = make_union(gpp, fncv3_selector)

    # Model
    xgb = XGBRegressor()
    cve = CrossValEstimator(estimator=xgb, cv=TimeSeriesSplit(n_splits=5))
    fn = FeatureNeutralizer(proportion=0.5)
    ens = NumeraiEnsemble()
    full_pipe = make_meta_pipeline(preproc_pipe, cve, ens, fn)

    # Train full model
    full_pipe.fit(X, y, numeraiensemble__eras=eras)

    # Inference
    preds = full_pipe.predict(X, eras=eras, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9 
    assert preds.shape[0] == X.shape[0]
    assert len(preds.shape) == 2

def test_multi_classification_ensemble(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)
    eras = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()
    preproc_pipe = make_union(GroupStatsPreProcessor(groups=['sunshine', 'rain']), ColumnSelector(fncv3_cols))

    model = DecisionTreeClassifier()
    crossval1 = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=3), predict_func='predict_proba')
    pred_rud = PredictionReducer(n_models=3, n_classes=5)
    ens2 = NumeraiEnsemble(donate_weighted=True)
    neut2 = FeatureNeutralizer(proportion=0.5)
    full_pipe = make_meta_pipeline(preproc_pipe, crossval1, pred_rud, ens2, neut2)

    y_int = (y * 4).astype(int)
    full_pipe.fit(X, y_int, numeraiensemble__eras=eras)

    preds = full_pipe.predict(X, eras=eras, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9 
    assert preds.shape[0] == X.shape[0]
    assert len(preds.shape) == 2

def test_feature_union_pipeline(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)
    eras = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()

    gpp = GroupStatsPreProcessor(groups=['sunshine', 'rain'])
    fncv3_selector = ColumnSelector(fncv3_cols)
    preproc_pipe = FeatureUnion([("gpp", gpp), ("selector", fncv3_selector)])

    xgb = MetaEstimator(XGBRegressor())
    fn = FeatureNeutralizer(proportion=0.5)
    model_pipe = make_pipeline(preproc_pipe, xgb, fn)

    model_pipe.fit(X, y)

    preds = model_pipe.predict(X, eras=eras, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9 
    assert preds.shape[0] == X.shape[0]

def test_column_transformer_pipeline(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)

    eras = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()

    gpp = GroupStatsPreProcessor(groups=['sunshine', 'rain'])
    preproc_pipe = ColumnTransformer([("gpp", gpp, features.columns.tolist()), 
                                      ("selector", "passthrough", fncv3_cols[2:])])
    xgb = MetaEstimator(XGBRegressor())
    fn = FeatureNeutralizer(proportion=0.5)
    model_pipe = make_pipeline(preproc_pipe, xgb, fn)

    model_pipe.fit(X, y)

    preds = model_pipe.predict(X, eras=eras, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9 
    assert preds.shape[0] == X.shape[0]

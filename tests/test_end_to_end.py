import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from sklego.preprocessing import ColumnSelector
from xgboost import XGBRegressor

from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.meta import CrossValEstimator, MetaEstimator, make_meta_pipeline
from numerblox.neutralizers import FeatureNeutralizer
from numerblox.numerframe import create_numerframe
from numerblox.preprocessing import GroupStatsPreProcessor


@pytest.fixture(scope="module")
def setup_data():
    df = create_numerframe("tests/test_assets/val_3_eras.parquet")
    return df


def test_neutralized_xgboost_pipeline(setup_data):
    df = setup_data

    X, y = df.get_feature_target_pair(multi_target=False)
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()
    era_series = df.get_era_data
    features = df.get_feature_data

    # Preprocessing
    gpp = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    fncv3_selector = ColumnSelector(fncv3_cols)
    # TODO Test with preproc FeatureUnion
    preproc_pipe = ColumnTransformer([("gpp", gpp, features.columns.tolist()), ("selector", fncv3_selector, fncv3_cols)])

    # Model
    xgb = XGBRegressor()
    cve = CrossValEstimator(estimator=xgb, cv=TimeSeriesSplit(n_splits=5))
    ens = NumeraiEnsemble()
    fn = FeatureNeutralizer(proportion=0.5)
    full_pipe = make_meta_pipeline(preproc_pipe, cve, ens, fn)

    # Train full model
    full_pipe.fit(X, y, era_series=era_series)
    # Inference
    preds = full_pipe.predict(X, era_series=era_series, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9
    assert preds.shape[0] == X.shape[0]
    assert len(preds.shape) == 2


def test_multi_classification_ensemble(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)
    era_series = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()
    # TODO Test with preproc FeatureUnion in sklearn 1.5+
    preproc_pipe = ColumnTransformer([("gpp", GroupStatsPreProcessor(groups=["sunshine", "rain"]), features.columns.tolist()), ("selector", ColumnSelector(fncv3_cols), fncv3_cols)])

    model = DecisionTreeClassifier()
    crossval = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=3), predict_func="predict_proba")
    pred_rud = PredictionReducer(n_models=3, n_classes=5)
    ens = NumeraiEnsemble(donate_weighted=True)
    fn = FeatureNeutralizer(proportion=0.5)
    full_pipe = make_meta_pipeline(preproc_pipe, crossval, pred_rud, ens, fn)

    y_int = (y * 4).astype(int)
    full_pipe.fit(X, y_int, era_series=era_series)

    preds = full_pipe.predict(X, era_series=era_series, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9
    assert preds.shape[0] == X.shape[0]
    assert len(preds.shape) == 2


@pytest.mark.xfail(reason="Can only be tested with sklearn 1.5+")
def test_feature_union_pipeline(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)
    era_series = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()

    gpp = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    fncv3_selector = ColumnSelector(fncv3_cols)
    preproc_pipe = make_union(gpp, fncv3_selector)

    xgb = MetaEstimator(XGBRegressor())
    fn = FeatureNeutralizer(proportion=0.5)
    model_pipe = make_pipeline(preproc_pipe, xgb, fn)

    model_pipe.fit(X, y)

    preds = model_pipe.predict(X, era_series=era_series, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9
    assert preds.shape[0] == X.shape[0]


def test_column_transformer_pipeline(setup_data):
    df = setup_data
    X, y = df.get_feature_target_pair(multi_target=False)

    era_series = df.get_era_data
    features = df.get_feature_data
    fncv3_cols = df.get_fncv3_feature_data.columns.tolist()

    gpp = GroupStatsPreProcessor(groups=["sunshine", "rain"])
    preproc_pipe = ColumnTransformer([("gpp", gpp, features.columns.tolist()), ("selector", "passthrough", fncv3_cols[2:])])
    xgb = MetaEstimator(XGBRegressor())
    fn = FeatureNeutralizer(proportion=0.5)
    model_pipe = make_pipeline(preproc_pipe, xgb, fn)

    model_pipe.fit(X, y)

    preds = model_pipe.predict(X, era_series=era_series, features=features)
    assert preds.min() >= 0
    assert abs(preds.max() - 1) <= 1e-9
    assert preds.shape[0] == X.shape[0]

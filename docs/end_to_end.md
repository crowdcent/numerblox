# End To End Examples

This section will show NumerBlox in action for some more advanced use cases. If you are looking for inspiration to leverage the power of NumerBlox, check out these examples.

First we download the classic data with NumeraiClassicDownloader. We use a NumerFrame for convenience to parse the dataset.

```py
from numerblox.numerframe import create_numerframe
from numerblox.download import NumeraiClassicDownloader
dl = NumeraiClassicDownloader(directory_path="my_numerai_data_folder")
dl.download_training_data("train_val", version="4.2", int8=True)
df = create_numerframe("my_numerai_data_folder/train_val/train_int8.parquet")
val_df = create_numerframe("my_numerai_data_folder/train_val/val_int8.parquet")

X, y = df.get_feature_target_pair(multi_target=False)
fncv3_cols = df.get_fncv3_feature_data.columns.tolist()

val_X, val_y = val_df.get_feature_target_pair(multi_target=False)
val_features = val_df.get_feature_data
val_eras = val_df.get_era_data
```

## 1. Neutralized XGBoost pipeline.

Let's construct an end-to-end pipeline that does the following:
- Augment FNCv3 features with group statistics features for the `sunshine` and `rain` data.
- Fit 5 folds of XGBoost.
- Ensemble them with a weighted average where the more recent folds get a higher weight.
- Neutralize the prediction with respect to the original features.

External libraries are xgboost and sklego. Make sure to have these dependencies installed.

```bash
!pip install xgboost sklego
```

```py
from xgboost import XGBRegressor
from sklego.preprocessing import ColumnSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer

from numerblox.preprocessing import GroupStatsPreProcessor
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble
from numerblox.neutralizers import FeatureNeutralizer

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
full_pipe.fit(X, y, numeraiensemble__eras=eras, featureneutralizer__eras=eras, featureneutralizer__features=features);

# Inference on validation data
val_preds = full_pipe.predict(val_X, eras=val_eras, features=val_features)
```

## 2. Multi Classification Ensemble

This example shows a multiclass classification example where the Numerai target is transformed into integers (`[0, 0.25, 0.5, 0.75, 1.0] -> [0, 1, 2, 3, 4]`) and treated as a classification problem. 

When we call `predict_proba` on a classifier the result will be a probability for every class, like for example `[0.1, 0.2, 0.3, 0.2, 0.2]`. In order to reduce these to one number we use the `PredictionReducer`, which takes the probabilities for every model and reduces it with a vector multiplication (Fro example, `[0.1, 0.2, 0.3, 0.2, 0.2] @ [0, 1, 2, 3, 4] = 2.2`). It does this for every model so the output of `PredictionReducer` has 3 columns. 

Because we set `donate_weighted=True` in `NumeraiEnsemble` 3 columns are reduced to one column using a weighted ensemble where the most recent fold get the highest weight. Lastly, the final prediction column is neutralized.

```py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.neutralizers import FeatureNeutralizer

model = DecisionTreeClassifier()
crossval1 = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=3), predict_func='predict_proba')
pred_rud = PredictionReducer(n_models=3, n_classes=5)
ens2 = NumeraiEnsemble(donate_weighted=True)
neut2 = FeatureNeutralizer(proportion=0.5)
full_pipe = make_meta_pipeline(preproc_pipe, crossval1, pred_rud, ens2, neut2)

full_pipe.fit(X, y, numeraiensemble__eras=eras, featureneutralizer__eras=eras, featureneutralizer__features=features)

preds = full_pipe.predict(val_X, eras=val_eras, features=val_features)
```

## 3. Ensemble of ensemble of regressors

This object introduces a `ColumnTransformer` that contains 3 pipelines. Each pipeline can have a different set of arguments. Here we simplify by passing every pipeline with the same columns. 
The output from all pipelines is concatenated, ensembled with `NumeraiEnsemble` and the final ensembles column is neutralized. Note that every fold here is equal weighted. If you want to give recent folds more weight set `weights` in `NumeraiEnsemble` for all `ColumnTransformer` output.

```py
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble,
from numerblox.neutralizers import FeatureNeutralizer

pipes = []
for i in range(3):
    model = DecisionTreeRegressor()
    crossval = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=5), predict_func='predict')
    pipe = make_pipeline(crossval)
    pipes.append(pipe)

models = make_column_transformer(*[(pipe, features.columns.tolist()) for pipe in pipes])
ens_end = NumeraiEnsemble()
neut = FeatureNeutralizer(proportion=0.5)
full_pipe = make_meta_pipeline(models, ens_end, neut)

full_pipe.fit(X, y, 
              columntransformer__eras=eras,
              numeraiensemble__eras=eras)

preds = full_pipe.predict(val_X, eras=val_eras, features=val_features)
```

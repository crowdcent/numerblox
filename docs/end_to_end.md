# End To End Examples

This section will show NumerBlox in action for some more advanced use cases. If you are looking for inspiration to really leverage the power of NumerBlox, check out these examples.




## 1. Numerai Classic: Single Target 5-fold weighted XGBoost with feature neutralization.

First we download the classic data with NumeraiClassicDownloader. We use a NumerFrame for convenience to parse the dataset.

```py
from numerblox.numerframe import create_numerframe
from numerblox.download import NumeraiClassicDownloader
dl = NumeraiClassicDownloader(directory_path="my_numerai_data_folder")
dl.download_training_data("train_val", version="4.2", int8=True)
df = create_numerframe("my_numerai_data_folder/train_val/train_int8.parquet")
val_df = create_numerframe("my_numerai_data_folder/train_val/val_int8.parquet")
```

Next let's construct an end-to-end pipeline that does the following:
- Augment FNCv3 features with group statistics features for the `sunshine` and `rain` data.
- Fit 5 folds of XGBoost.
- Ensemble them with a weighted average where the more recent folds get a higher weight.
- Neutralize the prediction with respect to the original features.

External libraries are xgboost and sklego. Make to have these dependencies installed.

```bash
!pip install xgboost sklego
```

```py
from xgboost import XGBRegressor
from sklego.preprocessing import ColumnSelector
from sklearn.pipeline import make_pipeline, make_union

from numerblox.preprocessing import GroupStatsPreProcessor
from numerblox.meta import CrossValEstimator
from numerblox.ensemble import NumeraiEnsemble
from numerblox.neutralizers import FeatureNeutralizer

X, y = df.get_feature_target_pair(multi_target=False)
fncv3_cols = df.get_fncv3_features.columns.tolist()

# Preprocessing
gpp = GroupStatsPreProcessor(groups=['sunshine', 'rain'])
fncv3_selector = ColumnSelector(fncv3_cols)

preproc_pipe = make_union(gpp, fncv3_selector)

# Model
xgb = XGBRegressor()
cve = CrossValEstimator(xgb, n_splits=5)
ens = NumeraiEnsemble(donate_weighted=True)
neut = FeatureNeutralizer(proportion=0.5)
model_pipe = make_pipeline(cve, ens, neut)

full_pipe = make_pipeline(preproc_pipe, model_pipe)

# Train full model
full_pipe.fit(X, y)

# Inference on validation data
val_X, val_y = val_df.get_feature_target_pair(multi_target=False)
val_preds = full_pipe.predict(val_X)
```

## 2. Multi Classification ensembling for multiple models






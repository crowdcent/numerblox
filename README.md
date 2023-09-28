![](https://img.shields.io/pypi/v/numerblox.png)
![](https://img.shields.io/pypi/pyversions/numerblox.png)
![](https://img.shields.io/github/contributors/crowdcent/numerblox.png)
![](https://img.shields.io/github/issues-raw/crowdcent/numerblox.png)
![](https://img.shields.io/codecov/c/github/crowdcent/numerblox.png)


# NumerBlox

NumerBlox offers components that help with developing strong Numerai models and inference pipelines. From downloading data to submitting predictions, NumerBlox has you covered.

All components can be used standalone and all processors are fully compatible to use within `scikit-learn` pipelines.  

**Documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)

## 1. Installation

### 1.1 Installation

Install numerblox from PyPi by running:

`pip install numerblox`

Alternatively you can clone this repository and install it in
development mode by installing using `poetry`:

```bash
git clone https://github.com/crowdcent/numerblox.git
pip install poetry
poetry install
```

### 1.2 Getting Started

Test your installation using one of the education notebooks in
`examples`. A good example is `numerframe_tutorial`. Run it in your
Notebook environment to quickly test if your installation has succeeded.
The documentation contains examples and explanations for each component of NumerBlox.

## 2. Core functionality

NumerBlox has the following features for both Numerai Classic and Signals:

1. Downloading data.
2. A custom data structure extending Pandas DataFrame (`NumerFrame``). It is not mandatory to use this data structure, but it simplifies parsing Numerai data, getting feature groups, targets, etc.
3. A suite of preprocessors.
4. Target engineering.
5. A suite of postprocessors (ensembling, neutralization and penalization)
6. A custom scikit-learn Pipeline (`MetaPipeline`) to fit postprocessors end-to-end with your preprocessing and models. This is only necessary if you want to use postprocessors in your pipeline.
7. A suite of meta-estimators like `CrossValEstimator` that allows you to fit multiple folds end-to-end in a scikit-learn pipeline.
8. A full evaluation suite with all metrics used by Numerai.
9. Submitters to easily and safely submit predictions.

Example notebooks for each of these components can be found in the `examples` directory. Also check out the documentation for more information.

**Full documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)


## 3. Quick Start

Below are two example of how NumerBlox can be used to train and do inference on Numerai data. For a full overview of all components check out the homepage of the documentation and its sections. For more advanced examples to leverage NumerBlox in your models check out the `End-To-End Examples` section in the documentation.

### 3.1 Simple example

The example below shows how NumerBlox simplifies training and inference on an XGBoost model.
NumerBlox is used here for easy downloading, data parsing, evaluation, inference and submission.

```python
import pandas as pd
from xgboost import XGBRegressor
from numerblox.misc import Key
from numerblox.numerframe import create_numerframe
from numerblox.download import NumeraiClassicDownloader
from numerblox.prediction_loaders import ExamplePredictions
from numerblox.evaluation import NumeraiClassicEvaluator
from numerblox.submission import NumeraiClassicSubmitter

# Download data
downloader = NumeraiClassicDownloader("data")
# Training and validation data
downloader.download_training_data("train_val", version="4.2", int8=True)
df = create_numerframe("data/train_val/train_int8.parquet")

# Train
X, y = df.get_feature_target_pair(multi_target=False)
xgb = XGBRegressor()
xgb.fit(X.values, y.values)

# Evaluate
val_df = create_numerframe("data/train_val/validation_int8.parquet")
val_df['prediction'] = xgb.predict(val_df.get_feature_data)
val_df['example_preds'] = ExamplePredictions("v4.2/validation_example_preds.parquet").fit_transform(None)['prediction'].values
evaluator = NumeraiClassicEvaluator()
metrics = evaluator.full_evaluation(val_df, 
                                    example_col="example_preds", 
                                    pred_cols=["prediction"], 
                                    target_col="target")

# Inference
downloader.download_inference_data("current_round", version="4.2", int8=True)
live_df = create_numerframe(file_path="data/current_round/live_int8.parquet")
live_X, live_y = live_df.get_feature_target_pair(multi_target=False)
preds = xgb.predict(live_X)

# Submit
NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"
key = Key(pub_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# Your prediction file with 'id' as index and defined 'cols' below.
pred_dataf = pd.DataFrame(preds, index=live_df.index, columns=["prediction"])
# Only works with valid key credentials and model_name
submitter.full_submission(dataf=pred_dataf,
                          cols="prediction",
                          file_name="submission.csv",
                          model_name="MY_MODEL_NAME")
```

### 3.2. Advanced NumerBlox modeling.

This example showcases how you can really push NumerBlox to create powerful pipelines. This pipeline approaches the Numerai Classic data as a classification problem. It fits multiple cross validation folds, reduces the classification probabilties to single values and create a weighted ensemble of these where the most recent folds get a higher weight. Lastly, the predictions are neutralized. The model is evaluated in validation data, inference is done on live data and a submission is done.
Lastly, we remove the download and submission directories to clean up the environment. This is especially convenient if you are running daily inference on your own server.

```py
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.prediction_loaders import ExamplePredictions
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.neutralizers import FeatureNeutralizer

# Download data
downloader = NumeraiClassicDownloader("data")
# Training and validation data
downloader.download_training_data("train_val", version="4.2", int8=True)
df = create_numerframe("data/train_val/train_int8.parquet")

# Setup model pipeline
model = XGBClassifier()
crossval1 = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=5), predict_func='predict_proba')
pred_rud = PredictionReducer(n_models=5, n_classes=5)
ens2 = NumeraiEnsemble(donate_weighted=True)
neut2 = FeatureNeutralizer(proportion=0.5)
full_pipe = make_meta_pipeline(preproc_pipe, crossval1, pred_rud, ens2, neut2)

# Train
X, y = df.get_feature_target_pair(multi_target=False)
y_int = (y * 4).astype(int)
eras = df.get_era_data
features = df.get_feature_data
full_pipe.fit(X.values, y_int.values, numeraiensemble__eras=eras, featureneutralizer__eras=eras, featureneutralizer__features=features)

# Evaluate
val_df = create_numerframe("data/train_val/validation_int8.parquet")
val_X, _ = val_df.get_feature_target_pair(multi_target=False)
val_eras = val_df.get_era_data
val_features = val_df.get_feature_data
val_df['prediction'] = full_pipe.predict(val_X, eras=val_eras, features=val_features)
val_df['example_preds'] = ExamplePredictions("v4.2/validation_example_preds.parquet").fit_transform(None)['prediction'].values
evaluator = NumeraiClassicEvaluator()
metrics = evaluator.full_evaluation(val_df, 
                                    example_col="example_preds", 
                                    pred_cols=["prediction"], 
                                    target_col="target")

# Inference
downloader.download_inference_data("current_round", version="4.2", int8=True)
live_df = create_numerframe(file_path="data/current_round/live_int8.parquet")
live_X, live_y = live_df.get_feature_target_pair(multi_target=False)
live_eras = live_df.get_era_data
live_features = live_df.get_feature_data
preds = full_pipe.predict(live_X, eras=live_eras, features=live_features)

# Submit
NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"
key = Key(pub_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# Your prediction file with 'id' as index and defined 'cols' below.
pred_dataf = pd.DataFrame(preds, index=live_df.index, columns=["prediction"])
# Only works with valid key credentials and model_name
submitter.full_submission(dataf=pred_dataf,
                          cols="prediction",
                          file_name="submission.csv",
                          model_name="MY_MODEL_NAME")

# Clean up environment
downloader.remove_base_directory()
submitter.remove_base_directory()
```


## 4. Contributing

Be sure to read the `How To Contribute` section in the documentation for detailed instructions on
contributing.

If you have questions or want to discuss new ideas for NumerBlox,
please create a Github issue first.

## 5. Crediting sources

Some of the components in this library may be based on forum posts,
notebooks or ideas made public by the Numerai community. We have done
our best to ask all parties who posted a specific piece of code for
their permission and credit their work in the documentation. If your
code is used in this library without credits, please let us know, so we
can add a link to your article/code.

If you are contributing to NumerBlox and are using ideas posted
earlier by someone else, make sure to credit them by posting a link to
their article/code in documentation.

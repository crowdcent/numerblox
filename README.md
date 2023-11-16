![](https://img.shields.io/pypi/v/numerblox.png)
![](https://img.shields.io/pypi/pyversions/numerblox.png)
![](https://img.shields.io/github/contributors/crowdcent/numerblox.png)
![](https://img.shields.io/codecov/c/gh/carlolepelaars/numerblox/master)
![](https://img.shields.io/pypi/dm/numerblox)


# NumerBlox

NumerBlox offers components that help with developing strong Numerai models and inference pipelines. From downloading data to submitting predictions, NumerBlox has you covered.

All components can be used standalone and all processors are fully compatible to use within [scikit-learn](https://scikit-learn.org/) pipelines.  

**Documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)

## 1. Installation

Install numerblox from PyPi by running:

`pip install numerblox`

Alternatively you can clone this repository and install it in
development mode by installing using `poetry`:

```bash
git clone https://github.com/crowdcent/numerblox.git
pip install poetry
cd numerblox
poetry install
```

Installation without dev dependencies can be done by adding `--only main` to the `poetry install` line.

Test your installation using one of the education notebooks in
[examples](https://github.com/crowdcent/numerblox/examples). Good places to start are [quickstart.ipynb](https://github.com/crowdcent/numerblox/examples/quickstart.ipynb) and [numerframe_tutorial.ipynb](https://github.com/crowdcent/numerblox/examples/numerframe_tutorial.ipynb). Run it in your
Notebook environment to quickly test if your installation has succeeded.
The documentation contains examples and explanations for each component of NumerBlox.

## 2. Core functionality

NumerBlox has the following features for both Numerai Classic and Signals:

**[Data Download](https://crowdcent.github.io/numerblox/download/):** Automated retrieval of Numerai datasets.

**[NumerFrame](https://crowdcent.github.io/numerblox/numerframe/):** A custom Pandas DataFrame for easier Numerai data manipulation.

**[Preprocessors](https://crowdcent.github.io/numerblox/preprocessing/):** Customizable techniques for data preprocessing.

**[Target Engineering](https://crowdcent.github.io/numerblox/targets/):** Tools for creating new target variables.

**[Postprocessors](https://crowdcent.github.io/numerblox/neutralization/):** Ensembling, neutralization, and penalization.

**[MetaPipeline](https://crowdcent.github.io/numerblox/meta/):** An era-aware pipeline extension of scikit-learn's Pipeline. Specifically designed to integrate with era-specific Postprocessors such as neutralization and ensembling. Can be optionally bypassed for custom implementations.

**[MetaEstimators](https://crowdcent.github.io/numerblox/meta/):** Era-aware estimators that extend scikit-learn's functionality. Includes features like CrossValEstimator which allow for era-specific, multiple-folds fitting seamlessly integrated into the pipeline.

**[Evaluation](https://crowdcent.github.io/numerblox/evaluation/):** Comprehensive metrics aligned with Numerai's evaluation criteria.

**[Submitters](https://crowdcent.github.io/numerblox/submission/):** Facilitates secure and easy submission of predictions.

Example notebooks for each of these components can be found in the [examples](https://github.com/crowdcent/numerblox/examples). Also check out [the documentation](https://crowdcent.github.io/numerblox) for more information.


## 3. Quick Start

Below are two examples of how NumerBlox can be used to train and do inference on Numerai data. For a full overview of all components check out the documentation. More advanced examples to leverage NumerBlox to the fullest can be found in the [End-To-End Example section](https://crowdcent.github.io/numerblox/end_to_end/).

### 3.1 Simple example

The example below shows how NumerBlox simplifies training and inference on an XGBoost model.
NumerBlox is used here for easy downloading, data parsing, evaluation, inference and submission. You can experiment with this setup yourself in the example notebook [quickstart.ipynb](https://github.com/crowdcent/numerblox/examples/quickstart.ipynb).

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

### 3.2. Advanced NumerBlox modeling

This example showcases how you can really push NumerBlox to create powerful pipelines. This pipeline approaches the Numerai Classic data as a classification problem. It fits multiple cross validation folds, reduces the classification probabilties to single values and create a weighted ensemble of these where the most recent folds get a higher weight. Lastly, the predictions are neutralized. The model is evaluated in validation data, inference is done on live data and a submission is done.
Lastly, we remove the download and submission directories to clean up the environment. This is especially convenient if you are running daily inference on your own server or a cloud VM.

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
crossval = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=5), predict_func='predict_proba')
pred_rud = PredictionReducer(n_models=5, n_classes=5)
ens = NumeraiEnsemble(donate_weighted=True)
neut = FeatureNeutralizer(proportion=0.5)
full_pipe = make_meta_pipeline(preproc_pipe, crossval, pred_rud, ens, neut)

# Train
X, y = df.get_feature_target_pair(multi_target=False)
y_int = (y * 4).astype(int)
eras = df.get_era_data
features = df.get_feature_data
full_pipe.fit(X, y_int, numeraiensemble__eras=eras)

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

Be sure to read the [How To Contribute section](https://crowdcent.github.io/numerblox/contributing/) section in the documentation for detailed instructions on
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

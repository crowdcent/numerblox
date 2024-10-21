![](https://img.shields.io/pypi/v/numerblox.png)
![Python Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/crowdcent/numerblox/master/pyproject.toml&query=%24.project%5B%22requires-python%22%5D&label=python&color=blue)
![](https://img.shields.io/github/contributors/crowdcent/numerblox.png)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://img.shields.io/codecov/c/gh/carlolepelaars/numerblox/master)
![](https://img.shields.io/pypi/dm/numerblox)




# NumerBlox

NumerBlox offers components that help with developing strong Numerai models and inference pipelines. From downloading data to submitting predictions, NumerBlox has you covered.

All components can be used standalone and all processors are fully compatible to use within [scikit-learn](https://scikit-learn.org/) pipelines.  

**Documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)

## 1. Installation

### Recommended (using pip)
Simply install numerblox from PyPI by running:

```bash
pip install numerblox
```

If you prefer to use [uv](https://github.com/astral-sh/uv), you can install numerblox with:

```bash
uv pip install numerblox
```

### Development
To install for development, clone the repository and use either pip or uv:

Using pip:
```bash
git clone https://github.com/crowdcent/numerblox.git
cd numerblox
pip install -e ".[test]"
```

Using [uv](https://github.com/astral-sh/uv):
```bash
git clone https://github.com/crowdcent/numerblox.git
cd numerblox
uv venv
uv pip install -e ".[test]"
```

For installation without dev dependencies, omit the `[test]` extra:

```bash
pip install -e .
```
or
```bash
uv pip install -e .
```

Test your installation using one of the education notebooks in
[examples](https://github.com/crowdcent/numerblox/examples). Good places to start are [quickstart.ipynb](https://github.com/crowdcent/numerblox/examples/quickstart.ipynb) and [numerframe_tutorial.ipynb](https://github.com/crowdcent/numerblox/examples/numerframe_tutorial.ipynb). Run it in your notebook environment to quickly test if your installation has succeeded. The documentation contains examples and explanations for each component of NumerBlox.

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

**[Model Upload](https://crowdcent.github.io/numerblox/model_upload/):** Assists in the process of uploading trained models to Numerai for automated submissions.

Example notebooks for each of these components can be found in the [examples](https://github.com/crowdcent/numerblox/examples). Also check out [the documentation](https://crowdcent.github.io/numerblox) for more information.


## 3. Quick Start

Below are two examples of how NumerBlox can be used to train and do inference on Numerai data. For a full overview of all components check out the documentation. More advanced examples to leverage NumerBlox to the fullest can be found in the [End-To-End Example section](https://crowdcent.github.io/numerblox/end_to_end/).

### 3.1. Simple example

The example below shows how NumerBlox can simplify the process of downloading, loading, training, evaluating, inferring and submitting data for Numerai Classic.

NumerBlox is used here for easy downloading, data parsing, evaluation, inference and submission. You can experiment with this setup yourself in the example notebook [quickstart.ipynb](https://github.com/crowdcent/numerblox/examples/quickstart.ipynb).

#### Downloading, loading, and training
```python
from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe
from xgboost import XGBRegressor

downloader = NumeraiClassicDownloader("data")
downloader.download_training_data("train_val", version="5.0")
df = create_numerframe("data/train_val/train.parquet")

X, y = df.get_feature_target_pair(multi_target=False)
model = XGBRegressor()
model.fit(X.values, y.values)
```

#### Evaluation
```python
from numerblox.prediction_loaders import ExamplePredictions
from numerblox.evaluation import NumeraiClassicEvaluator

val_df = create_numerframe("data/train_val/validation.parquet")
val_df['prediction'] = model.predict(val_df.get_feature_data)
val_df['example_preds'] = ExamplePredictions("v5.0/validation_example_preds.parquet").fit_transform(None)['prediction'].values
evaluator = NumeraiClassicEvaluator()
metrics = evaluator.full_evaluation(val_df, 
                                    example_col="example_preds", 
                                    pred_cols=["prediction"], 
                                    target_col="target")
```

#### Live Inference
```python
downloader.download_live_data("current_round", version="5.0")
live_df = create_numerframe(file_path="data/current_round/live.parquet")
live_X, live_y = live_df.get_feature_target_pair(multi_target=False)
preds = model.predict(live_X)
```

#### Submission
```python
from numerblox.misc import Key
from numerblox.submission import NumeraiClassicSubmitter

NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"
key = Key(pub_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
pred_dataf = pd.DataFrame(preds, index=live_df.index, columns=["prediction"])
submitter.full_submission(dataf=pred_dataf,
                          cols="prediction",
                          file_name="submission.csv",
                          model_name="MY_MODEL_NAME")
```

#### Model Upload
```python
from numerblox.submission import NumeraiModelUpload

uploader = NumeraiModelUpload(key=key, max_retries=3, sleep_time=15, fail_silently=True)
uploader.create_and_upload_model(model=model, 
                                 model_name="MY_MODEL_NAME", 
                                 file_path="models/my_model.pkl")
```

### 3.2. Advanced NumerBlox modeling

Building on the simple example, this advanced setup showcases how to leverage NumerBlox's powerful components to create a sophisticated pipeline that can replace the "simple" XGBoost model in the example above. This advanced example creates an extensible scikit-learn pipeline with metadata routing that:

- Approaches Numerai Classic as a classification problem
- Uses cross-validation with multiple folds
- Reduces classification probabilities to single values
- Creates a weighted ensemble favoring recent folds
- Applies neutralization to the predictions

#### Creating the pipeline
```python
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.neutralizers import FeatureNeutralizer

model = XGBClassifier()
crossval = CrossValEstimator(estimator=model, cv=TimeSeriesSplit(n_splits=5), predict_func='predict_proba')
pred_rud = PredictionReducer(n_models=5, n_classes=5)
ens = NumeraiEnsemble(donate_weighted=True)
neut = FeatureNeutralizer(proportion=0.5)
full_pipe = make_meta_pipeline(crossval, pred_rud, ens, neut)
```

#### Training
```python
# ... Assume df is already defined as in the simple example ...
X, y = df.get_feature_target_pair(multi_target=False)
y_int = (y * 4).astype(int)  # Convert targets to integer classes for classification
era_series = df.get_era_data
features = df.get_feature_data
full_pipe.fit(X, y_int, era_series=era_series)
```

#### Inference
```python
live_eras = live_df.get_era_data
live_features = live_df.get_feature_data
preds = full_pipe.predict(live_X, era_series=live_eras, features=live_features)
```

Scikit-learn estimators, pipelines, and metadata routing are used to make sure we pass the correct era and feature information to estimators in the pipeline that require those parameters. It is worth familiarizing yourself with these concepts before using the advanced modeling features of NumerBlox: 
- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn metadata routing](https://scikit-learn.org/stable/metadata_routing.html)

## 4. Contributing

Be sure to read the [How To Contribute section](https://crowdcent.github.io/numerblox/contributing/) section in the documentation for detailed instructions on contributing.

If you have questions or want to discuss new ideas for NumerBlox, please create a Github issue first.

## 5. Crediting sources

Some of the components in this library may be based on forum posts, notebooks or ideas made public by the Numerai community. We have done our best to ask all parties who posted a specific piece of code for their permission and credit their work in docstrings and documentation. If your code is public and used in this library without credits, please let us know, so we can add a link to your article/code. We want to always give credit where credit is due.

If you are contributing to NumerBlox and are using ideas posted earlier by someone else, make sure to credit them by posting a link to their article/code in docstrings and documentation.

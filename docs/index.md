# NumerBlox

NumerBlox offers components that help with developing strong Numerai models and inference pipelines. From downloading data to submitting predictions, NumerBlox has you covered.

All components can be used standalone and all processors are fully compatible to use within `scikit-learn` pipelines.  

**Documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)

![](https://img.shields.io/pypi/v/numerblox.png)
![](https://img.shields.io/pypi/pyversions/numerblox.png)
![](https://img.shields.io/github/contributors/crowdcent/numerblox.png)
![](https://img.shields.io/github/issues-raw/crowdcent/numerblox.png)
![](https://img.shields.io/github/repo-size/crowdcent/numerblox.png)
![](https://img.shields.io/codecov/c/github/crowdcent/numerblox.png)


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

Example notebooks for each of these components can be found in the `examples` directory.

**Full documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)


## 3. Examples

Below we will illustrate some common use cases in NumerBlox. To
learn more in-depth about the features of this library, check out
notebooks in `examples`.

### 3.1. Downloading Numerai Classic Data

`NumeraiClassicDownloader` allows you to download just the data you need with a few lines of code and handles the directory structure for you. All data from v4+ is supported. For Numerai Signals we provide downloaders from several sources for which you can find more information in the documentation under the `Downloaders` section.

```py
import pandas as pd
from numerblox.download import NumeraiClassicDownloader

downloader = NumeraiClassicDownloader("data")
# Training and validation data
downloader.download_training_data("train_val", version="4.2", int8=True)
# Live data
downloader.download_inference_data("current_round", version="4.2", int8=True)
df = pd.read_parquet(file_path="data/current_round/live.parquet")
```

### 3.2. Core NumerFrame features

NumerFrame is powerful data structure which simplifies working with Numerai data. Below are a few examples of how you can leverage NumerFrame for your Numerai workflow. Under the hood NumerFrame is a Pandas DataFrame so you still have access to all Pandas functionality when using NumerFrame.

NumerFrame usage is completely optional. Other NumerBlox components do not depend on it, though they are compatible with it.


```py
from numerblox.numerframe import create_numerframe

df = create_numerframe(file_path="data/current_round/live.parquet")
# Get data for features, targets and predictions
features = df.get_feature_data
targets = df.get_target_data
predictions = df.get_prediction_data

# Get specific data groups
fncv3_features = df.get_fncv3_features
group_features = df.get_group_features(group='rain')

# Fetch columns by pattern. For example all 20 day targets.
pattern_data = df.get_pattern_data(pattern='_20')
# Or for example Jerome targets.
jerome_targets = df.get_pattern_data(pattern='_jerome_')

# Split into feature and target pairs. Will get single target by default.
X, y = df.get_feature_target_pair()
# Optionally get all targets
X, y = df.get_feature_target_pair(multi_target=True)

# Fetch data for specified eras
X, y = df.get_era_batch(eras=['0001', '0002'])

# Since every operation returns a NumerFrame they can be chained.
# An example chained operation is getting features and targets for the last 2 eras.
X, y = df.get_last_eras(2).get_feature_target_pair()
```

### 3.3. Advanced Numerai models

All core processors in `numerblox` are compatible with `scikit-learn` and therefore also `scikit-learn` extension libraries like [scikit-lego](https://github.com/koaning/scikit-lego), [umap](https://github.com/lmcinnes/umap) and [scikit-llm](https://github.com/iryna-kondr/scikit-llm). 

The example below illustrates its seamless integration with `scikit-learn`. Aside from core `scikit-learn` processors we use `ColumnSelector` from the [scikit-lego](https://github.com/koaning/scikit-lego) extension library.

For more examples check out the notebooks in the `examples` directory and the `End-To-End Examples` section in the documentation.

```py
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import make_union
from sklearn.model_selection import TimeSeriesSplit
from sklego.preprocessing import ColumnSelector

from numerblox.meta import CrossValEstimator
from numerblox.preprocessing import GroupStatsPreProcessor
from numerblox.numerframe import create_numerframe

# Easy data parsing with NumerFrame
df = create_numerframe(file_path="data/train_val/train_int8.parquet")
val_df = create_numerframe(file_path="data/train_val/validation_int8.parquet")

X, y = df.get_feature_target_pair()
eras = df.get_era_data()

val_X, val_y = val_df.get_feature_target_pair()
val_eras = val_df.get_era_data()

fncv3_cols = nf.get_fncv3_features.columns.tolist()

# Sunshine/Rain group statistics and FNCv3 features as model input
gpp = GroupStatsPreProcessor(groups=['sunshine', 'rain'])
fncv3_selector = ColumnSelector(fncv3_cols)
preproc_pipe = make_union(gpp, fncv3_selector)

# 5 fold cross validation with XGBoost as model
model = CrossValEstimator(XGBRegressor(), cv=TimeSeriesSplit(n_splits=5))
# Ensemble 5 folds with weighted average
ensembler = NumeraiEnsemble(donate_weighted=True)

full_pipe = make_pipeline(preproc_pipe, model, ensembler)

full_pipe.fit(X, y, numeraiensemble__eras=eras)

val_preds = full_pipe.predict(val_X, eras=val_eras)
```

### 3.4. Evaluation

`NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator` take care of computing all evaluation metrics for you. Below is a quick example of using it for Numerai Classic. 
For more information on advanced usage and which metrics are computed check the `Evaluators` section in the documentation.

```py
from numerblox.evaluation import NumeraiClassicEvaluator

# Validation DataFrame to compute metrics on
val_df = ...

evaluator = NumeraiClassicEvaluator()
metrics = evaluator.full_evaluation(val_df, 
                                    era_col="era", 
                                    example_col="example_preds", 
                                    pred_cols=["prediction"], 
                                    target_col="target")
```


### 3.5. Submission

Submission for both Numerai Class and Signals can be done with a few lines of code. Here we illustrate an example for Numerai Classic. Check out the `Submitters` section in the documentation for more information.

```py
from numerblox.misc import Key
from numerblox.submission import NumeraiClassicSubmitter

NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"

# Your predictions on the live data
predictions = ...

# Fill in you public and secret key for Numerai
key = Key(pub_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# full_submission checks contents, saves as csv and submits.
submitter.full_submission(dataf=predictions,
                          cols="prediction",
                          model_name="YOUR_MODEL_NAME")
# (optional) Clean up directory after submission
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

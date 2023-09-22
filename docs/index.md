# NumerBlox

`numerblox` offers components that can be used across the Numerai process. From downloading data to submitting predictions, `numerblox` has you covered.

All components can be used stand alone and are fully compatible to use within `scikit-learn` pipelines.  

**Documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox)

![](https://img.shields.io/pypi/v/numerblox.png)
![](https://img.shields.io/pypi/pyversions/numerblox.png)
![](https://img.shields.io/github/contributors/crowdcent/numerblox.png)
![](https://img.shields.io/github/issues-raw/crowdcent/numerblox.png)
![](https://img.shields.io/github/repo-size/crowdcent/numerblox.png)
![](https://img.shields.io/codecov/c/github/crowdcent/numerblox.png)


## 1. Install

## 1. Getting Started

### 1.1 Installation

Install numerblox from PyPi by running:

`pip install numerblox`

Alternatively you can clone this repository and install it in
development mode running the following from the root of the repository:

`pip install -e .`

### 1.2 Running Notebooks

Test your installation using one of the education notebooks in
`nbs/edu_nbs`. A good example is `numerframe_tutorial`. Run it in your
Notebook environment to quickly test if your installation has succeeded.

### 2.1. Contents

#### 2.1.1. Core functionality

`numerblox` features the following functionality for both Numerai Classic and Signals:

1. Downloading data.
2. A custom data structure extending Pandas DataFrame (NumerFrame). It is not mandatory to use this data structure, but it simplifies getting feature groups, targets, etc.
3. A suite of preprocessors.
4. Target engineering.
5. A suite of postprocessors (ensembling, neutralization and penalization)
6. A custom scikit-learn Pipeline (`MetaPipeline`) to fit postprocessors end-to-end with your preprocessing and models.
7. A suite of meta-estimators like `CrossValEstimator` that allows you to fit multiple folds end-to-end in a scikit-learn pipeline.
8. A full evaluation suite with all metrics used by Numerai.
9. Submitting predictions.

Example notebooks can be found in the `examples` directory.

**Full documentation:**
[crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox/)

### 2.2. Examples

Below we will illustrate a common use case for inference pipelines. To
learn more in-depth about the features of this library, check out
notebooks in `examples`.

#### 2.2.1. Numerai Classic

``` python
# --- 0. Numerblox dependencies ---
from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe
from numerblox.key import Key
from numerblox.submission import NumeraiClassicSubmitter

# --- 1. Download version 4.2 data ---
downloader = NumeraiClassicDownloader("data")
downloader.download_inference_data("current_round")

# --- 2. Initialize NumerFrame (optional) ---
dataf = create_numerframe(file_path="data/current_round/live.parquet")

# --- 3. Define and run pipeline. ---
# All numerblox pre- and postprocessors are compatible with scikit-learn
result_dataf = ...

# --- 4. Submit ---
# Set credentials
key = Key(pub_id="Hello", secret_key="World")
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# full_submission checks contents, saves as csv and submits.
submitter.full_submission(dataf=result_dataf,
                          cols=f"prediction_test_neutralized_0.5",
                          model_name="test")

# --- 5. Clean up environment (optional) ---
downloader.remove_base_directory()
submitter.remove_base_directory()
```

#### 2.2.2. Numerai Signals

``` python
# --- 0. Numerblox dependencies ---
from numerblox.download import KaggleDownloader
from numerblox.numerframe import create_numerframe
from numerblox.key import Key
from numerblox.submission import NumeraiSignalsSubmitter

# --- 1. Download Katsu1110 yfinance dataset from Kaggle ---
kd = KaggleDownloader("data")
kd.download_inference_data("code1110/yfinance-stock-price-data-for-numerai-signals")

# --- 2. Initialize NumerFrame (optional) ---
dataf = create_numerframe("data/full_data.parquet")

# --- 3. Define and run pipeline ---
# All numerblox pre- and postprocessors are compatible with scikit-learn
result_dataf = ...

# --- 4. Submit ---
# Set credentials
key = Key(pub_id="Hello", secret_key="World")
submitter = NumeraiSignalsSubmitter(directory_path="sub_current_round", key=key)
# full_submission checks contents, saves as csv and submits.
# cols selection must at least contain 1 ticker column and a signal column.
submitter.full_submission(dataf=result_dataf,
                          cols=['bloomberg_ticker', 'signal'],
                          model_name="test_model1")

# --- 5. Clean up environment (optional) ---
kd.remove_base_directory()
submitter.remove_base_directory()
```

## 3. Contributing

Be sure to read `contributing.md` for detailed instructions on
contributing.

If you have questions or want to discuss new ideas for `numerblox`,
please create a Github issue first.

## 4. Crediting sources

Some of the components in this library may be based on forum posts,
notebooks or ideas made public by the Numerai community. We have done
our best to ask all parties who posted a specific piece of code for
their permission and credit their work in the documentation. If your
code is used in this library without credits, please let us know, so we
can add a link to your article/code.

If you are contributing to `numerblox` and are using ideas posted
earlier by someone else, make sure to credit them by posting a link to
their article/code in documentation.

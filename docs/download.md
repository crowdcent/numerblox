# Downloaders

## Numerai Classic

`NumeraiClassicDownloader` simplifies downloading of datasets from Numerai's API. It allows you to easily download data with a few lines and the data is automatically organized in directories.

NOTE: Only int8 versions are available in this downloader. From v4.2 onwards, Numerai only provides int8 versions of the data.

```py
from numerblox.download import NumeraiClassicDownloader

dl = NumeraiClassicDownloader(directory_path="my_numerai_data_folder")

# Training and validation data
dl.download_training_data("train_val", version="4.3")

# Inference data (live)
dl.download_inference_data("live", version="4.3")
```

Besides these common use cases you can also get feature sets and meta model predictions with `NumeraiClassicDownloader`. 

```py
from numerblox.download import NumeraiClassicDownloader

dl = NumeraiClassicDownloader(directory_path="my_numerai_data_folder")

# Get feature sets (v4.3)
features = dl.get_classic_features()

# Get meta model predictions
dl.download_meta_model_preds()
meta_model_preds = pd.read_parquet("my_numerai_data_folder/meta_model.parquet")
```

## Numerai Signals

For Numerai Signals we have several options implemented to download data. Numerai Signals Downloaders have similar methods as `NumeraiClassicDownloader`.

### EOD Historical Data

Download data from EOD historical data. A common data vendor used for Numerai Signals. 

More information: [https://eodhistoricaldata.com](https://eodhistoricaldata.com)


Make sure you have the underlying Python package for EOD installed.

```bash
pip install eod
```

For EOD you also need to define credentials in the form of an API key.

More information: [https://eodhd.com/pricing](https://eodhd.com/pricing)

```py
from numerblox.download import EODDownloader

eod_api_key = "MY_EOD_API_KEY"
tickers = ["AAPL.US", "MSFT.US", "GOOG.US"]
dl = EODDownloader(directory_path="my_numerai_signals_folder",
key=eod_api_key, tickers=tickers)

# Download full dataset
dl.download_training_data(start="2008-01-01")

# load data directly into DataFrame from January 1st 2023 for inference.
live_data = dl.downloader_live_data(start="2023-01-01")
```

### Kaggle

Some Numerai dataset are uploaded and maintained on Kaggle Datasets. NumerBlox offers a convenient API to download these datasets.

For authentication, make sure you have a directory called .kaggle in your home directory
with therein a kaggle.json file. kaggle.json should have the following structure:
`{"username": USERNAME, "key": KAGGLE_API_KEY}`

More info on authentication: [github.com/Kaggle/kaggle-api#api-credentials](https://github.com/Kaggle/kaggle-api#api-credentials)

More info on the Kaggle Python API: [kaggle.com/donkeys/kaggle-python-api](https://kaggle.com/donkeys/kaggle-python-api)

Also make sure you have the `kaggle` Python package installed.

```bash
pip install kaggle
```

Below is a quickstart example using Katsu's starter dataset.

```py
from numerblox.download import KaggleDownloader

kd = KaggleDownloader(directory_path="my_numerai_signals_folder")

# A good example of Numerai Signals data on Kaggle Datasets is Katsu1110's yfinance price dataset.
kd.download_inference_data("code1110/yfinance-stock-price-data-for-numerai-signals")
```

### Rolling your own downloader

We invite users to build out their own downloaders for Numerai Signals. The only requirements are that you inherit from `numerblox.download.BaseDownloader` and implement the `download_training_data` and `download_inference_data` methods. Below you will find a template for this.

If you have a downloader that you would like to share with the community, please open a Pull Request in NumerBlox.

```py
class AwesomeCustomDownloader(BaseDownloader):
    """
    TEMPLATE -
    Download awesome financial data for Numerai Signals from who knows where.

    :param directory_path: Base folder to download files to.
    """
    def __init__(self, directory_path: str):
        super().__init__(directory_path=directory_path)

    def download_inference_data(self, *args, **kwargs):
        """ (minimal) weekly inference downloading here. """
        ...

    def download_training_data(self, *args, **kwargs):
        """ Training + validation dataset downloading here. """
        ...

```








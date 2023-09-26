# Preprocessors

NumerBlox offers a suite of preprocessors to easily do Numerai specific data transformations. All preprocessors are compatible with `scikit-learn` pipelines and feature a similar API. Note that some preprocessors may require an additional `eras` or `tickers` argument in the `transform` step.

## Numerai Classic

### GroupStatsPreProcessor

The v4.2 (rain) dataset for Numerai Classic reintroduced feature groups. The `GroupStatsPreProcessor` calculates group statistics for all data groups. It uses predefined feature group mappings to generate statistical measures (mean, standard deviation, skew) for each of the feature groups. 

#### Example

Here's how you can use the `GroupStatsPreProcessor`:

```python
from numerblox.preprocessing import GroupStatsPreProcessor
group_processor = GroupStatsPreProcessor(groups=['intelligence'])

# Return features with group statistics for the 'intelligence' group
features = group_processor.transform(X)
```

## Numerai Signals

### ReduceMemoryProcessor

The `ReduceMemoryProcessor` reduces the memory usage of the data as much as possible. It's particularly useful for Numerai Signals dataset which can be quite large.

Note that modern Numerai Classic Data (v4.2+) already is an int8 format so this processor will be not be useful for Numerai Classic.

```py
from numerblox.preprocessing import ReduceMemoryProcessor

processor = ReduceMemoryProcessor(deep_mem_inspect=True, verbose=True)
reduced_data = processor.fit_transform(dataf)
```

### KatsuFeatureGenerator

`KatsuFeatureGenerator` performs feature engineering based on [Katsu's starter notebook](https://www.kaggle.com/code1110/numeraisignals-starter-for-beginners). This is useful for those participating in the Numerai Signals contest.

You can specify custom windows that indicates how many days to look back when generating features.

```py
from numerblox.preprocessing import KatsuFeatureGenerator

feature_gen = KatsuFeatureGenerator(windows=[7, 14, 21])
enhanced_data = feature_gen.fit_transform(dataf)
```

### EraQuantileProcessor

`EraQuantileProcessor` transforms features into quantiles by era. This can help normalize data and make patterns more distinguishable.

Using `.transform` requires passing the era column as a `pd.Series`. This is because the quantiles are calculated per era so it needs that information along with the raw input features.

```py
from numerblox.preprocessing import EraQuantileProcessor

eq_processor = EraQuantileProcessor(num_quantiles=50, random_state=42)
transformed_data = eq_processor.fit_transform(X, eras=eras_series)
```

### TickerMapper

`TickerMapper` maps tickers from one format to another. Useful when working with data from multiple sources that have different ticker formats.

For the default ticker mapper the following formats are supported: `['ticker', 'bloomberg_ticker', 'yahoo']`. You can also specify a custom mapping by passing a dictionary to the `mapper_path` argument at the instantiation.

```py
from numerblox.preprocessing import TickerMapper

# Transform from ticker to Bloomberg format.
ticker_mapper = TickerMapper(ticker_col="ticker", target_ticker_format="bloomberg_ticker")
mapped_data = ticker_mapper.transform(dataf["ticker"])
```

### LagPreProcessor

`LagPreProcessor` generates lag features based on specified windows. Lag features can capture temporal patterns in time-series data.

Note that `LagPreProcessor` needs a series of `tickers` in the `.transform` step.

```py
from numerblox.preprocessing import LagPreProcessor

lag_processor = LagPreProcessor(windows=[5, 10, 20])
lag_processor.fit(X)
lagged_data = lag_processor.transform(X, tickers=tickers_series)

```

### DifferencePreProcessor

`DifferencePreProcessor` computes the difference between features and their lags. It's used after `LagPreProcessor`.

WARNING: `DifferencePreProcessor` works only on `pd.DataFrame` and with columns that are generated in `LagPreProcessor`. If you are using these in a Pipeline make sure `LagPreProcessor` is defined before `DifferencePreProcessor` and that output API is set to Pandas (`pipeline.set_output(transform="pandas")`).

Note that `LagPreProcessor` needs a series of `tickers` in the `.transform` step so a pipeline with both preprocessors will need a `tickers` argument in `.transform`.

```py
from sklearn.pipeline import make_pipeline
from numerblox.preprocessing import DifferencePreProcessor

lag = LagPreProcessor(windows=[5, 10])
diff = DifferencePreProcessor(windows=[5, 10], pct_diff=True)
pipe = make_pipeline(lag, diff)
pipe.set_output(transform="pandas")
pipe.fit(X)
diff_data = pipe.transform(X, tickers=tickers_series)
```

### PandasTaFeatureGenerator

`PandasTaFeatureGenerator` uses the `pandas-ta` library to generate technical analysis features. It's a powerful tool for those interested in financial time-series data.

Make sure you have `pandas-ta` installed before using this feature generator:

```bash
!pip install pandas-ta
```

Currently `PandasTaFeatureGenerator` only works on `pd.DataFrame` input. Its input is a DataFrame with columns `[ticker, date, open, high, low, close, volume]`.

```py
from numerblox.preprocessing import PandasTaFeatureGenerator

ta_gen = PandasTaFeatureGenerator()
ta_features = ta_gen.transform(dataf)
```

## Rolling your own preprocessor

We invite the community to contribute their own preprocessors to NumerBlox. If you have a preprocessor that you think would be useful to others, please open a PR with your code and tests.
The new preprocessor should adhere to [scikit-learn conventions](https://scikit-learn.org/stable/developers/develop.html). Here are some the most important things to keep in mind and a template.

- Make sure that your preprocessor inherits from `numerblox.preprocessing.base.BasePreProcessor`. This will automatically implement a blank fit method. It will also inherit from `sklearn.base.TransformerMixin` and `sklearn.base.BaseEstimator`.
- Make sure your preprocessor implements a `transform` method that can take a `np.array` or `pd.DataFrame` as input and outputs an `np.array`. If your preprocessor can only work with `pd.DataFrame` input, mention this explicitly in the docstring.
- Implement a `get_feature_names_out` method so it can support `pd.DataFrame` output with valid column names.

```py
import numpy as np
import pandas as pd
from typing import Union
from sklearn.validation import check_is_fitted, check_X_y
from numerblox.preprocessing.base import BasePreProcessor

class MyAwesomePreProcessor(BasePreProcessor):
    def __init__(self, random_state: int = 0):
        super().__init__()
        # If you introduce additional arguments be sure to add them as attributes.
        self.random_state = random_state

    def fit(self, X: Union[np.array, pd.DataFrame], y=None):
        # Arguments can be set for later use.
        self.n_cols_ = X.shape[1]
        return self

    def transform(self, X: Union[np.array, pd.DataFrame]) -> np.array:
        # Do your preprocessing here.
        # Can involve additional checks.
        check_is_fitted(self)
        X = check_X_y(X)
        return X

    def get_feature_names_out(self, input_features=None) -> list:
        # Return a list of feature names.
        # If you are not using pandas output, you can skip this method.
        check_is_fitted(self)
        return ["awesome_output_feature_{i}" for i in range(self.n_cols_)]
```

# NumerFrame

`NumerFrame` is an extension of `pd.DataFrame` tailored specifically for the data format and workflow commonly used by Numerai participants. It builds upon the base functionalities of a Pandas DataFrame by offering utilities that simplify working with Numerai datasets.

## Why?
- **Intuitive Data Handling**: With built-in features like `get_feature_data`, `get_target_data`, and more, it simplifies extracting data subsets specific to Numerai competitions.
  
- **Automated Column Grouping**: Automatically parses columns into recognizable groups such as features, targets, predictions, making data retrieval more intuitive and less error-prone.
  
- **Support for Multiple Formats**: Through `create_numerframe`, it supports initializing from various data formats such as CSV, Parquet, Excel, and Pickle, providing a flexible interface for users.
  
- **Optimized for Numerai**: Whether you're trying to fetch specific eras, feature groups or patterns like all 20-day targets, `NumerFrame` is designed to simplify those tasks for Numerai participants.
  
- **Chainable Operations**: Since most operations return another `NumerFrame`, they can be conveniently chained for more complex workflows.
  
- **Tailored for Machine Learning**: With methods like `get_feature_target_pair`, it aids in easily splitting the data for machine learning tasks specific to the Numerai competition.
  
By using `NumerFrame`, participants can focus more on model development and less on data wrangling, leading to a smoother and more efficient workflow in the Numerai competition.


## Initialization
A NumerFrame can be initialized either from an existing `pd.DataFrame` or with `create_numerframe`. The `create_numerframe` function takes a path to a file and returns a `NumerFrame` object. This function automatically parses the file and supports CSV, Parquet, Excel and Pickle formats.

`NumerFrame` automatically parses columns into groups so you can easily retrieve what you need. It automatically is aware of the `era` column for its operations. 

`NumerFrame` follows a convention for feature groups.

- Features are all columns that start with `feature`.

- Targets are all columns that start with `target`.

- Predictions are all columns that start with `prediction`.

- Aux columns are all that fall in none of these buckets, like `era`, `data_type` and `id`. 

- Era column is either `era` or `date`.

```py
import pandas as pd
from numerblox.numerframe import NumerFrame, create_numerframe
# From DataFrame
data = pd.read_parquet('train.parquet')
df = NumerFrame(data)

# With create_numerframe
df = create_numerframe('train.parquet')
```


## Examples

Basic functionality: 
```py
# Get data for features, targets, predictions, and aux
features = df.get_feature_data
targets = df.get_target_data
predictions = df.get_prediction_data
aux_data = df.get_aux_data
```

Additionally it is possible to get groups specific to Numerai Classic like FNCv3 and internal feature groups. The examples below show some advanced functionality in `NumerFrame`.

```py
# Get data for features, targets and predictions
features = df.get_feature_data
targets = df.get_target_data
predictions = df.get_prediction_data

# Get specific data groups
fncv3_features = df.get_fncv3_feature_data
group_features = df.get_group_features(group='rain')
small_features = df.get_small_feature_data
medium_features = df.get_medium_feature_data

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


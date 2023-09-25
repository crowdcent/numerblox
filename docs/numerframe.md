# NumerFrame

`NumerFrame` is an extension of `pd.DataFrame` tailored for Numerai data. Easily get features, targets, predictions and/or eras with a single line of code.

### Initialization:
A NumerFrame can be initialized either from an existing `pd.DataFrame`` or with `create_numerframe`. The `create_numerframe` function takes a path to a file and returns a `NumerFrame` object. This function automatically parses the file and supports CSV, Parquet, Excel and Pickle formats.
```py
# From DataFrame
data = pd.read_parquet('train.parquet')
nf = NumerFrame(data)

# With create_numerframe
nf = create_numerframe('train.parquet')
```

`NumerFrame` automatically parses columns into groups so you can easily retrieve what you need. It automatically is aware of the `era` column for its operations. Possible options for the `era` columns are `era`, `date` and `friday_date`.

### Core Properties:
- **feature_cols**: Columns starting with "feature".
- **target_cols**: Columns starting with "target".
- **prediction_cols**: Columns starting with "prediction".
- **aux_cols**: All other columns.

All the groups can be retrieved with one line. 
```py
# Get data for features, targets, predictions, and aux
features = nf.get_feature_data
targets = nf.get_target_data
predictions = nf.get_prediction_data
aux_data = nf.get_aux_data
```

Additionally it is possible to get groups specific to Numerai Classic like FNCv3 and internal feature groups.
```py
# Get specific data groups
fncv3_features = nf.get_fncv3_features
group_features = np.get_group_features(group='rain')
```

### Key Methods & Usage:
```py
# Fetch columns by pattern. For example all 20 day targets.
pattern_data = nf.get_pattern_data(pattern='_20')
# Or for example Jerome targets.
jerome_targets = nf.get_pattern_data(pattern='_jerome_')

# Split into feature and target pairs. Will get single target by default.
X, y = nf.get_feature_target_pair()
# Optionally get all targets
X, y = nf.get_feature_target_pair(multi_target=True)

# Fetch data for specified eras
X, y = nf.get_era_batch(eras=['era1', 'era2'])
# Optionally get Tensorflow tensors for NN training
X, y = nf.get_era_batch(eras=['era1', 'era2'], convert_to_tf=True)

# Since every operation returns a NumerFrame they can be chained.
# Like for example getting features and targets for the last 2 eras.
X, y = nf.get_last_eras(2).get_feature_target_pair()
```

# Meta Estimators

Meta estimator wrap existing scikit-learn estimators to provide additional functionality. Currently, the following meta estimators are available:

- [CrossValEstimator](#crossvalestimator)
- [MetaPipeline](#metapipeline)

## CrossValEstimator

`CrossValEstimator` provides a way to integrate cross-validation directly into model training, enabling simultaneous fitting of multiple models across data folds. By doing this, you can fit it as one transformer and get outputs for each fold during the prediction phase.

### Why CrossValEstimator?

- **Holistic Training**: Cross-validation offers a more robust model training process by leveraging multiple sub-sets of your data. This way, your model's performance is less susceptible to the peculiarities of any single data split.

- **Inherent Ensemble**: By training on multiple folds, you're essentially building an ensemble of models. Ensembles often outperform individual models since they average out biases, reduce variance, and are less likely to overfit.

- **Custom Evaluation**: With the `evaluation_func` parameter, you can input your custom evaluation logic, allowing for flexible and tailored performance assessment for each fold.

- **Flexibility with Predictions**: Choose between different prediction functions like 'predict', 'predict_proba', and 'predict_log_proba' using the `predict_func` parameter.

- **Verbose Logging**: Gain insights into the training process with detailed logs during the fitting phase, aiding in debugging and understanding model performance across folds.

### Example

```py
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from numerblox.meta import CrossValEstimator

# Define the cross-validation strategy
cv = KFold(n_splits=5)

# Initialize the estimator
estimator = XGBRegressor(n_estimators=100, max_depth=3)

# (optional) Define a custom evaluation function
def custom_eval(y_true, y_pred):
    return {"mse": ((y_true - y_pred) ** 2).mean()}

# Initialize the CrossValEstimator
cross_val_estimator = CrossValEstimator(cv=cv, 
                                        estimator=estimator,
                                        evaluation_func=custom_eval)

# Fit the CrossValEstimator
cross_val_estimator.fit(X_train, y_train)
predictions = cross_val_estimator.predict(X_test)
```

## MetaPipeline

The `MetaPipeline` extends the functionality of scikit-learn's `Pipeline` by seamlessly integrating models and post-model transformations. It empowers you to employ sophisticated data transformation techniques not just before, but also after your model's predictions. This is particularly useful when post-processing predictions, such as neutralizing feature exposures in financial models.

## Why MetaPipeline?

- **Post-Model Transformations**: It can be crucial to apply transformations, like feature neutralization, after obtaining predictions. `MetaPipeline` facilitates such operations, leading to improved model generalization and stability.

- **Streamlined Workflow**: Instead of managing separate sequences for transformations and predictions, you can orchestrate them under a single umbrella, simplifying both development and production workflows.

- **Flexible Integration**: `MetaPipeline` gracefully handles a variety of objects, including `Pipeline`, `FeatureUnion`, and `ColumnTransformer`. This makes it a versatile tool adaptable to diverse tasks and data structures.

#### Example

Consider a scenario where you have an `XGBRegressor` model and want to apply a `FeatureNeutralizer` after obtaining the model's predictions:

```py
from xgboost import XGBRegressor
from numerblox.meta import MetaPipeline 
from numerblox.neutralizers import FeatureNeutralizer

# Define MetaPipeline steps
steps = [
    ('xgb_regressor', XGBRegressor(n_estimators=100, max_depth=3)),
    ('feature_neutralizer', FeatureNeutralizer(proportion=0.5))
]

# Create MetaPipeline
meta_pipeline = MetaPipeline(steps)

# Train and predict using MetaPipeline
meta_pipeline.fit(X_train, y_train)
predictions = meta_pipeline.predict(X_test)
```

For a more succinct creation of a `MetaPipeline`, you can use the `make_meta_pipeline` function:

```py
from numerblox.meta import make_meta_pipeline

pipeline = make_meta_pipeline(XGBRegressor(n_estimators=100, max_depth=3),
                              FeatureNeutralizer(proportion=0.5))
```
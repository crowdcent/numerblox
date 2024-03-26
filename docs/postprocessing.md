# Postprocessing

## Feature Neutralization

`FeatureNeutralizer` provides classic feature neutralization by subtracting linear model influence, ensuring that predictions are not overly influenced by a specific set of features.

### Why?
- **Reduce Overfitting**: By neutralizing predictions, you can potentially reduce the risk of overfitting to specific feature characteristics.
- **Control Feature Influence**: Allows you to have a granular control on how much influence a set of features can exert on the final predictions. 
- **Enhance Model Robustness**: By limiting the influence of potentially noisy or unstable features, you might improve the robustness of your model's predictions across different data periods.

### Quickstart

Make sure to pass both the features to use for penalization as a `pd.DataFrame` and the accompanying era column as a `pd.Series` to the `predict` method.

Additionally, `pred_name` and `proportion` can be lists. In this case, the neutralization will be performed for each prediction name and proportion. For example, if `pred_name=["prediction1", "prediction2"]` and `proportion=[0.5, 0.7]`, then the result will be an array with 4 neutralized prediction columns.
All neutralizations will be performed in parallel.

Single column neutralization:
```python
import sklearn
import pandas as pd
from numerblox.neutralizers import FeatureNeutralizer

# Enable sklearn custom arguments (i.e. metadata routing)
sklearn.set_config(enable_metadata_routing=True)

predictions = pd.Series([0.24, 0.87, 0.6])
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

neutralizer = FeatureNeutralizer(pred_name="prediction", proportion=0.5)
neutralizer.set_predict_request(era_series=True, features=True)
neutralizer.fit()
neutralized_predictions = neutralizer.predict(X=predictions, features=feature_data, eras=era_data)
```

Multiple column neutralization:
```python
import sklearn
import pandas as pd
from numerblox.neutralizers import FeatureNeutralizer

# Enable sklearn custom arguments (i.e. metadata routing)
sklearn.set_config(enable_metadata_routing=True)

predictions = pd.DataFrame({"prediction1": [0.24, 0.87, 0.6], "prediction2": [0.24, 0.87, 0.6]})
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

neutralizer = FeatureNeutralizer(pred_name=["prediction1", "prediction2"], proportion=[0.5, 0.7])
neutralizer.set_predict_request(era_series=True, features=True)
neutralizer.fit()
neutralized_predictions = neutralizer.predict(X=predictions, features=feature_data, eras=era_data)
```

## FeaturePenalizer

`FeaturePenalizer` neutralizes predictions using TensorFlow based on provided feature exposures. It's designed to integrate seamlessly with scikit-learn.

### Why?
- **Limit Feature Exposure**: Ensures that predictions are not excessively influenced by any individual feature, which can help in achieving more stable predictions.
- **Enhanced Prediction Stability**: By penalizing high feature exposures, it might lead to more stable and consistent predictions across different eras or data splits.
- **Mitigate Model Biases**: If a model is relying too heavily on a particular feature, penalizing can help in balancing out the biases and making the model more generalizable.

### Quickstart

Make sure to pass both the features to use for penalization as a `pd.DataFrame` and the accompanying era column as a `pd.Series` to the `predict` method.
```python
import sklearn
from numerblox.penalizers import FeaturePenalizer

# Enable sklearn custom arguments (i.e. metadata routing)
sklearn.set_config(enable_metadata_routing=True)

predictions = pd.Series([0.24, 0.87, 0.6])
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

penalizer = FeaturePenalizer(max_exposure=0.1, pred_name="prediction")
penalizer.set_predict_request(era_series=True, features=True)
penalizer.fit(X=predictions)
penalized_predictions = penalizer.predict(X=predictions, features=feature_data, eras=era_data)
```

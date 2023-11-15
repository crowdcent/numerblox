# Feature Neutralization

`FeatureNeutralizer` provides classic feature neutralization by subtracting a linear model's influence, ensuring that predictions are not overly influenced by a specific set of features.

## Why?
- **Reduce Overfitting**: By neutralizing predictions, you can potentially reduce the risk of overfitting to specific feature characteristics.
- **Control Feature Influence**: Allows you to have a granular control on how much influence a set of features can exert on the final predictions. 
- **Enhance Model Robustness**: By limiting the influence of potentially noisy or unstable features, you might improve the robustness of your model's predictions across different data periods.

## Quickstart

Make sure to pass both the features to use for penalization as a `pd.DataFrame` and the accompanying era column as a `pd.Series` to the `predict` method.
```python
import pandas as pd
from numerblox.neutralizers import FeatureNeutralizer

predictions = pd.Series([0.24, 0.87, 0.6])
feature_data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
era_data = pd.Series([1, 1, 2])

neutralizer = FeatureNeutralizer(pred_name="prediction", proportion=0.5, cuda=False)
neutralizer.fit()
neutralized_predictions = neutralizer.predict(X=predictions, features=feature_data, eras=era_data)
```

## Note
Neutralization can be run on the GPU by setting `cuda=True`. When setting this ensure you have CuPy installed.

```bash
pip install cupy
```

# Models

## EraBoostedXGBRegressor

`EraBoostedXGBRegressor` is a custom regressor extending the functionality of XGBoost, aimed at improving accuracy on specific eras in a dataset. It upweights the eras that are toughest to fit. It is designed to integrate seamlessly with scikit-learn.

### Why?
- Era-Specific Focus: Targets the worst-performing eras in your data for performance enhancement, ensuring that the model improves where it is most needed.
- Scikit-learn integration: `EraBoostedXGBRegressor` is designed to integrate seamlessly with scikit-learn.
- Customization Options: Offers flexibility to adjust the proportion of eras to focus on, the number of trees added per iteration, and the total number of iterations for era boosting.

### Quickstart

Make sure to include the era column as a `pd.Series` in the `fit` method.
```python
from numerblox.models import EraBoostedXGBRegressor

model = EraBoostedXGBRegressor(proportion=0.5, trees_per_step=10, num_iters=20)
model.fit(X=X_train, y=y_train, era_series=eras_train)

predictions = model.predict(X_live)
```
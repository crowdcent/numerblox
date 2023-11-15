# Prediction Loaders

Prediction loaders are designed to seamlessly fetch and transform prediction data, especially from Numerai's API. These classes can be integrated into pipelines to automate the prediction generation process for the Numerai competition.

# Why?

Numerai provides example predictions to help participants understand the expected structure and format of predictions. With the ExamplePredictions class, you can easily fetch these example predictions for different data versions, allowing you to quickly evaluate or test your models against the Numerai's standard prediction dataset.

# ExamplePredictions

## Usage:
The `ExamplePredictions` class fetches the example predictions for the specified version of the Numerai dataset. This can be useful for testing or understanding the prediction structure and data distribution.

```py
from numerblox.prediction_loaders import ExamplePredictions
# Instantiate and load example predictions for v4.2
example_loader = ExamplePredictions(file_name="v4.2/live_example_preds.parquet")
example_preds_df = example_loader.transform()
```

Besides the v4.2 data you can also retrieve example preds from earlier datasets. Check [Numerai's data page](https://numer.ai/data) to see which datasets are supported.

```py
from numerblox.prediction_loaders import ExamplePredictions
example_loader_v41 = ExamplePredictions(file_name="v4.1/live_example_preds.parquet")
example_preds_v41_df = example_loader_v41.transform()
```

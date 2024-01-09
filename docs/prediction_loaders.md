# Prediction Loaders

Prediction loaders are designed to seamlessly fetch and transform prediction data, especially from Numerai's API. These classes can be integrated into pipelines to automate the prediction generation process for the Numerai competition.

# Why?

Numerai provides example predictions to help participants understand the expected structure and format of predictions. With the ExamplePredictions class, you can easily fetch these example predictions for different data versions, allowing you to quickly evaluate or test your models against the Numerai's standard prediction dataset.

# ExamplePredictions

## Usage:
The `ExamplePredictions` class fetches the example predictions for the specified version of the Numerai dataset. This can be useful for testing or understanding the prediction structure and data distribution.

Downloaded files are automatically cleaned up after data is loaded with the `transform` method. To keep the files make sure to set `keep_files=True` when instantiating the class.

```py
from numerblox.prediction_loaders import ExamplePredictions
# Instantiate and load example predictions for v4.3
example_loader = ExamplePredictions(file_name="v4.3/live_example_preds.parquet", keep_files=False)
example_preds_df = example_loader.transform()
```

Besides the v4.3 data you can also retrieve example preds from earlier datasets. Check [Numerai's data page](https://numer.ai/data) to see which datasets are supported.

```py
from numerblox.prediction_loaders import ExamplePredictions
example_loader_v42 = ExamplePredictions(file_name="v4.2/live_example_preds.parquet", keep_files=True)
example_preds_v42_df = example_loader_v41.transform()
```

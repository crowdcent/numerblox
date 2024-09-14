
# Numerai Model Upload

The `NumeraiModelUpload` class is designed for uploading pickled models to Numerai for automated submissions. It supports both single models and ensembles with multiple stacking levels, allowing for flexibility in model architecture. This class is particularly useful for automating the submission process while handling model serialization, validation, and uploading seamlessly.

## Why Use NumeraiModelUpload?

- **Automation**: Automates the model submission process to Numerai, reducing the need for manual intervention.
- **Support for Multi-Layer Ensembles**: Allows for sophisticated model stacking and ensembling by providing trained models and weights for each layer, enabling users to submit highly customized models.
- **Error Handling**: Offers robust error handling with retry logic, ensuring reliable uploads even in case of network or API errors.
- **Custom Predict Function**: Supports custom prediction functions for advanced use cases, offering greater flexibility.

## Instantiation

To use `NumeraiModelUpload`, instantiate it with a directory path to store models, a `Key` object containing your credentials, and optional parameters for error handling.

```python
from numerblox.misc import Key
from numerblox.submission import NumeraiModelUpload

key = Key(pub_id="your_public_id", secret_key="your_secret_key")

uploader = NumeraiModelUpload(
    directory_path="models/",
    key=key,
    max_retries=3,
    sleep_time=15,
    fail_silently=True
)
```

### Parameters:

- **`directory_path`**: (str, optional) Directory path to store and read models. Defaults to an empty string.
- **`key`**: (Key) Key object containing valid credentials for Numerai Classic.
- **`max_retries`**: (int, optional) Maximum number of retries for uploading models to Numerai. Defaults to 2.
- **`sleep_time`**: (int, optional) Time in seconds to wait between retries. Defaults to 10.
- **`fail_silently`**: (bool, optional) Whether to suppress errors and skip failed uploads without raising exceptions. Useful for batch processing. Defaults to `False`.
- **`*args, **kwargs`**: Additional arguments passed to `NumerAPI` initialization.

## Model Uploading

The primary method for uploading models is `create_and_upload_model`, which serializes the model using `cloudpickle`, saves it to a file, and uploads it to Numerai.

### Example: Upload a Single Model

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Assuming you have a trained model named 'rf_model'
rf_model = RandomForestRegressor()

uploader.create_and_upload_model(
    model=rf_model,
    feature_cols=['feature_1', 'feature_2'],
    model_name="my_model_name",
    file_path="models/my_model.pkl"
)
```

### Method: `create_and_upload_model`

Wraps the model or ensemble into a predict function, serializes it with `cloudpickle`, saves it to a file, and uploads it to Numerai.

#### Parameters:

- **`model`**: (Union[Any, Dict[int, Dict[str, Any]]]) Trained model object or a dictionary specifying ensemble layers. If a dictionary, format should be `{layer_number: {'models': [list of model objects], 'weights': [list of weights (optional)]}}`.
- **`feature_cols`**: (Optional[List[str]]) List of feature column names. If `None`, all columns starting with "feature_" will be used. Ignored if the model is a dictionary (ensemble).
- **`model_name`**: (str) Numerai model name.
- **`file_path`**: (str) Full path where the pickle file will be saved.
- **`data_version`**: (Optional[str]) Data version ID or name.
- **`docker_image`**: (Optional[str]) Docker image ID or name.
- **`custom_predict_func`**: (Optional[Callable[[pd.DataFrame], pd.DataFrame]]) Custom predict function. If provided, it should accept a DataFrame and return a DataFrame with a "prediction" column. Ignored if the model is a dictionary (ensemble).

#### Returns:

- **`upload_id`**: Model upload ID if successful.

### Method: `get_available_data_versions`

Retrieve available data versions for model uploads.

#### Example

```python
available_data_versions = uploader.get_available_data_versions()
print(available_data_versions)
```

### Method: `get_available_docker_images`

Retrieve available Docker images for model uploads.

#### Example

```python
available_docker_images = uploader.get_available_docker_images()
print(available_docker_images)
```

### Method: `_get_model_id`

Private method to get the model ID needed for model uploading.

#### Parameters:

- **`model_name`**: (str) The name of the model registered in Numerai.

#### Returns:

- **`model_id`**: (str) Corresponding model ID for the given model name.

### Method: `get_model_mapping`

Property that returns a mapping between raw model names and their corresponding model IDs.

#### Example

```python
model_mapping = uploader.get_model_mapping
print(model_mapping)
```

## Example: Upload an Ensemble Model

To upload an ensemble model with multiple layers, use the following approach:

```python
ensemble_spec = {
    1: {
        'models': [rf_model1, rf_model2],
        'weights': [0.6, 0.4]
    },
    2: {
        'models': [meta_model],
        'weights': [1.0]
    }
}

uploader.create_and_upload_model(
    model=ensemble_spec,
    model_name="ensemble_model_name",
    file_path="models/ensemble_model.pkl"
)
```

## Note

Ensure that the credentials and model names used in the above examples match those configured in your Numerai account.
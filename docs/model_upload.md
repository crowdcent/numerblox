# Numerai Model Upload

The `NumeraiModelUpload` class is designed for uploading trained models to Numerai for automated submissions. You can upload a single trained model or a complete `sklearn` pipeline, allowing seamless integration with various machine learning workflows. This class efficiently handles model serialization, validation, and uploading, making it adaptable for different types of models and workflows.

***Warning**: The `NumeraiModelUpload` class is designed to work with very specific requirements. For compatibility, make sure your environment matches the requirements listed in the official numerai-predict repository: [numerai-predict/requirements.txt](https://github.com/numerai/numerai-predict/blob/master/requirements.txt). Using different versions or additional packages may lead to issues during model upload and execution.*

## Why Use NumeraiModelUpload?

- **Automation**: Automates the model submission process to Numerai, reducing the need for manual intervention.
- **Support for Sklearn Pipelines**: Integrates seamlessly with `sklearn` pipelines and NumerBlox processors, allowing users to submit models with preprocessing, feature engineering, and stacking in a single workflow.
- **Error Handling**: Offers robust error handling with retry logic, ensuring reliable uploads even in case of network or API errors.
- **Custom Predict Function**: Supports custom prediction functions for advanced use cases, offering greater flexibility.

## Instantiation

To use `NumeraiModelUpload`, instantiate it with a `Key` object containing your credentials and optional parameters for error handling.

```python
from numerblox.misc import Key
from numerblox.submission import NumeraiModelUpload

key = Key(pub_id="your_public_id", secret_key="your_secret_key")

uploader = NumeraiModelUpload(
    key=key,
    max_retries=3,
    sleep_time=15,
    fail_silently=True
)
```

### Parameters:

- **`key`**: (Key) Key object containing valid credentials for Numerai Classic.
- **`max_retries`**: (int, optional) Maximum number of retries for uploading models to Numerai. Defaults to 2.
- **`sleep_time`**: (int, optional) Time in seconds to wait between retries. Defaults to 10.
- **`fail_silently`**: (bool, optional) Whether to suppress errors and skip failed uploads without raising exceptions. Useful for batch processing. Defaults to `False`.
- **`*args, **kwargs`**: Additional arguments passed to `NumerAPI` initialization.

## Model Uploading

The primary method for uploading models is `create_and_upload_model`, which serializes the model using `cloudpickle`, saves it to a file, and uploads it to Numerai.

### Example: Upload a Single Model

```python
import pandas as pd
from some_ml_library import TrainedModel

# Assume you have a trained model named 'my_model'
my_model = TrainedModel()

uploader.create_and_upload_model(
    model=my_model,
    model_name="my_model_name",
    file_path="models/my_model.pkl"
)
```

### Method: `create_and_upload_model`

Creates a model prediction function, serializes it, and uploads the model to Numerai.

#### Parameters:

- **`model`**: (Any) The machine learning model object.
- **`feature_cols`**: (Optional[List[str]]) List of feature column names for predictions. If `None`, all columns starting with "feature_" will be used.
- **`model_name`**: (str) Numerai model name.
- **`file_path`**: (str) Full path where the serialized model function will be saved.
- **`data_version`**: (Optional[str]) Data version to use for model upload.
- **`docker_image`**: (Optional[str]) Docker image to use for model upload.
- **`custom_predict_func`**: (Optional[Callable[[pd.DataFrame], pd.DataFrame]]) Custom predict function. If provided, it should accept a DataFrame and return a DataFrame with a "prediction" column.

#### Returns:

- **`upload_id`**: Upload ID if successful, `None` otherwise.

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

## Example: Upload an Ensemble Model with Sklearn Pipeline

To upload an ensemble model with multiple layers using an `sklearn` pipeline:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

# Create base models
base_models = [
    ('rf', RandomForestRegressor()),
    ('ridge', RidgeCV())
]

# Create stacking ensemble model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor())

uploader.create_and_upload_model(
    model=stacking_model,
    model_name="ensemble_model_name",
    file_path="models/ensemble_model.pkl"
)
```

## Note

Ensure that the credentials and model names used in the above examples match those configured in your Numerai account.

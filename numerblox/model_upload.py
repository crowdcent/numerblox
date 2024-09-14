import time
from typing import List, Callable, Optional, Union, Dict, Any
import pandas as pd
import cloudpickle
from numerapi import NumerAPI
from .download import BaseIO
from .misc import Key
import numpy as np

class NumeraiModelUpload(BaseIO):
    """
    Class for uploading pickled models to Numerai for automated submissions.
    Supports ensembles of models with multiple stacking levels.

    :param directory_path: Directory to store and read models
    :param key: Key object containing valid credentials for Numerai Classic.
    :param max_retries: Maximum number of retries for uploading models to Numerai.
    :param sleep_time: Time to sleep between uploading retries.
    :param fail_silently: Whether to skip uploading to Numerai without raising an error.
    Useful if you are uploading many models in a loop and want to skip models that fail to upload.
    *args, **kwargs will be passed to NumerAPI initialization.
    """

    def __init__(self, directory_path: str = '', key: Key = None,
                 max_retries: int = 2, sleep_time: int = 10,
                 fail_silently: bool = False, *args, **kwargs):
        super().__init__(directory_path)
        self.api = NumerAPI(public_id=key.pub_id, secret_key=key.secret_key, *args, **kwargs)
        self.max_retries = max_retries
        self.sleep_time = sleep_time
        self.fail_silently = fail_silently

    def create_and_upload_model(self,
                                model: Union[Any, Dict[int, Dict[str, Any]]],
                                feature_cols: Optional[List[str]] = None,
                                model_name: str = None,
                                file_path: str = None,
                                data_version: str = None,
                                docker_image: str = None,
                                custom_predict_func: Callable[[pd.DataFrame], pd.DataFrame] = None):
        """
        Wrap the model or ensemble into a predict function, serialize it with cloudpickle,
        save it to a user-specified file, and upload to Numerai.

        :param model: Trained model object, or a dictionary specifying ensemble layers.
                      If a dictionary, format should be {layer_number: {'models': [list of model objects],
                                                                      'weights': [list of weights (optional)]}}.
        :param feature_cols: List of feature column names. If None, all columns starting with "feature_" will be used.
                             Ignored if model is a dictionary (ensemble).
        :param model_name: Numerai model name.
        :param file_path: Full path where the pickle file will be saved.
        :param data_version: Data version ID or name (optional).
        :param docker_image: Docker image ID or name (optional).
        :param custom_predict_func: Custom predict function (optional). If provided, it should
                                    accept a DataFrame and return a DataFrame with a "prediction" column.
                                    Ignored if model is a dictionary (ensemble).
        :returns: Model upload ID if successful.
        """
        if custom_predict_func is not None:
            # Use the custom predict function provided
            predict = custom_predict_func
        elif isinstance(model, dict):
            # Handle ensemble with layers specified in dict
            ensemble_model = EnsembleModel(model)
            # Use the ensemble_model's predict function
            def predict(live_features: pd.DataFrame) -> pd.DataFrame:
                predictions = ensemble_model.predict(live_features)
                return pd.DataFrame({"prediction": predictions}, index=live_features.index)
        else:
            # Handle single model in the standard way
            def predict(live_features: pd.DataFrame) -> pd.DataFrame:
                if feature_cols is None:
                    # Select all columns starting with "feature_"
                    feature_cols_local = [col for col in live_features.columns if col.startswith("feature_")]
                else:
                    feature_cols_local = feature_cols
                live_predictions = model.predict(live_features[feature_cols_local])
                submission = pd.Series(live_predictions, index=live_features.index).rank(pct=True, method='first')
                return submission.to_frame("prediction")

        # Serialize the predict function using cloudpickle
        print(f"Serializing the predict function and saving to '{file_path}'")
        with open(file_path, "wb") as f:
            cloudpickle.dump(predict, f)

        # Upload the pickle file to Numerai
        model_id = self._get_model_id(model_name=model_name)
        api_type = self.api.__class__.__name__
        print(f"{api_type}: Uploading model from '{file_path}' for model '{model_name}' (model_id='{model_id}')")
        for attempt in range(self.max_retries):
            try:
                upload_id = self.api.model_upload(
                    file_path=file_path,
                    model_id=model_id,
                    data_version=data_version,
                    docker_image=docker_image
                )
                print(f"{api_type} model upload of '{file_path}' for '{model_name}' is successful! Upload ID: {upload_id}")
                return upload_id
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Failed to upload model '{file_path}' for '{model_name}' to Numerai. Retrying in {self.sleep_time} seconds...")
                    print(f"Error: {e}")
                    time.sleep(self.sleep_time)
                else:
                    if self.fail_silently:
                        print(f"Failed to upload model '{file_path}' for '{model_name}' to Numerai. Skipping...")
                        print(f"Error: {e}")
                    else:
                        print(f"Failed to upload model '{file_path}' for '{model_name}' after {self.max_retries} attempts.")
                        raise e

    def get_available_data_versions(self) -> dict:
        """Retrieve available data versions for model uploads."""
        return self.api.model_upload_data_versions()

    def get_available_docker_images(self) -> dict:
        """Retrieve available Docker images for model uploads."""
        return self.api.model_upload_docker_images()

    def _get_model_id(self, model_name: str) -> str:
        """Get model ID needed for model uploading."""
        model_mapping = self.get_model_mapping
        if model_name in model_mapping:
            return model_mapping[model_name]
        else:
            available_models = ', '.join(model_mapping.keys())
            raise ValueError(f"Model name '{model_name}' not found in your Numerai account. "
                             f"Available model names: {available_models}")

    @property
    def get_model_mapping(self) -> dict:
        """Mapping between raw model names and model IDs."""
        return self.api.get_models()


class EnsembleModel:
    """
    Class for handling ensembles of models, including multiple stacking levels.

    :param ensemble_spec: Dictionary specifying ensemble layers and weights.
                          Format: {layer_number: {'models': [list of model objects],
                                                  'weights': [list of weights (optional)]}}
    """

    def __init__(self, ensemble_spec: Dict[int, Dict[str, Any]]):
        if not isinstance(ensemble_spec, dict):
            raise ValueError("Ensemble specification must be a dictionary.")
        self.layers = self._validate_and_prepare_layers(ensemble_spec)

    def _validate_and_prepare_layers(self, ensemble_dict):
        """Validate the ensemble_dict and prepare the layers."""
        layers = {}
        for layer_num, layer_info in sorted(ensemble_dict.items()):
            if not isinstance(layer_num, int) or layer_num < 1:
                raise ValueError("Layer numbers must be integers starting from 1.")
            models = layer_info.get('models')
            weights = layer_info.get('weights', None)
            if not models or not isinstance(models, list):
                raise ValueError(f"Layer {layer_num}: 'models' must be a non-empty list of model objects.")
            if weights is not None:
                if not isinstance(weights, list) or len(weights) != len(models):
                    raise ValueError(f"Layer {layer_num}: 'weights' must be a list of the same length as 'models'.")
                if not abs(sum(weights) - 1.0) < 1e-6:
                    raise ValueError(f"Layer {layer_num}: Weights must sum to 1.")
            else:
                # Assign equal weights if not provided
                weights = [1 / len(models)] * len(models)
            layers[layer_num] = {
                'models': models,
                'weights': weights
            }
        if 1 not in layers:
            raise ValueError("At least one layer (layer 1) must be specified.")
        return layers

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the ensemble of models."""
        layer_outputs = {}
        max_layer = max(self.layers.keys())
        for layer_num in range(1, max_layer + 1):
            layer = self.layers[layer_num]
            models = layer['models']
            weights = layer['weights']
            predictions = []
            for weight, model in zip(weights, models):
                # Determine feature columns
                if layer_num == 1:
                    # Use original features for the first layer
                    features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                        [col for col in X.columns if col.startswith("feature_")]
                    input_data = X[features]
                else:
                    # Use output from the previous layer as input
                    input_data = layer_outputs[layer_num - 1]

                # Make predictions
                pred = model.predict(input_data)
                if isinstance(pred, np.ndarray):
                    pred = pd.Series(pred, index=X.index)
                pred = pred * weight
                predictions.append(pred)

            # Combine predictions for this layer into a DataFrame
            combined_pred = pd.concat(predictions, axis=1)

            # Convert combined predictions to DataFrame to ensure correct input format for the next layer
            if layer_num < max_layer:
                layer_outputs[layer_num] = combined_pred  # Pass as DataFrame to next layer
            else:
                # For the final layer, produce a ranked Series output
                final_prediction = combined_pred.mean(axis=1).rank(pct=True, method='first')
                return final_prediction

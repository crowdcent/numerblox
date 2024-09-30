import time
from typing import Any, Callable, List, Optional, Union

import cloudpickle
import pandas as pd
from numerapi import NumerAPI

from .misc import Key


class NumeraiModelUpload:
    """
    A class to handle the uploading of machine learning models to Numerai's servers.

    :param key: API key object containing public and secret keys for NumerAPI authentication.
    :param max_retries: Maximum number of attempts to upload the model.
    :param sleep_time: Number of seconds to wait between retries.
    :param fail_silently: Whether to suppress exceptions during upload.
    """

    def __init__(self, key: Key = None, max_retries: int = 2, sleep_time: int = 10, fail_silently: bool = False, *args, **kwargs):
        """
        Initializes the NumeraiModelUpload class with the necessary configuration.

        :param key: API key object containing public and secret keys for NumerAPI authentication.
        :param max_retries: Maximum number of retry attempts for model upload.
        :param sleep_time: Time (in seconds) to wait between retries.
        :param fail_silently: If True, suppress errors during model upload.
        :param *args: Additional arguments for NumerAPI.
        :param **kwargs: Additional keyword arguments for NumerAPI.
        """
        # Initialize NumerAPI with the provided keys and other arguments
        self.api = NumerAPI(public_id=key.pub_id, secret_key=key.secret_key, *args, **kwargs)
        self.max_retries = max_retries  # Set the maximum number of retries
        self.sleep_time = sleep_time  # Set the sleep time between retries
        self.fail_silently = fail_silently  # Determine whether to fail silently

    def create_and_upload_model(self, model: Any, feature_cols: Optional[List[str]] = None, model_name: str = None, file_path: str = None, data_version: str = None, docker_image: str = None, custom_predict_func: Callable[[pd.DataFrame], pd.DataFrame] = None) -> Union[str, None]:
        """
        Creates a model prediction function, serializes it, and uploads the model to Numerai.
        :param model: The machine learning model object.
        :param feature_cols: List of feature column names for predictions. Defaults to None.
        :param model_name: The name of the model to upload.
        :param file_path: The file path where the serialized model function will be saved.
        :param data_version: Data version to use for model upload.
        :param docker_image: Docker image to use for model upload.
        :param custom_predict_func: Custom prediction function to use instead of the model's predict method.

        :return: Upload ID if the upload is successful, None otherwise.
        """
        # Determine which prediction function to use
        if custom_predict_func is not None:
            predict = custom_predict_func  # Use custom prediction function if provided
        else:
            # Define default prediction function
            def predict(live_features: pd.DataFrame) -> pd.DataFrame:
                # Determine feature columns to use for predictions
                if feature_cols is None:
                    feature_cols_local = [col for col in live_features.columns if col.startswith("feature_")]
                else:
                    feature_cols_local = feature_cols

                # Predict using the model
                live_predictions = model.predict(live_features[feature_cols_local])

                # Rank predictions and convert to a DataFrame
                submission = pd.Series(live_predictions, index=live_features.index).rank(pct=True, method="first")
                return submission.to_frame("prediction")

        # Serialize the prediction function and save to the specified file path
        print(f"Serializing the predict function and saving to '{file_path}'")
        with open(file_path, "wb") as f:
            cloudpickle.dump(predict, f)

        # Get the model ID for the specified model name
        model_id = self._get_model_id(model_name=model_name)
        api_type = self.api.__class__.__name__  # Get the type of API being used
        print(f"{api_type}: Uploading model from '{file_path}' for model '{model_name}' (model_id='{model_id}')")

        # Attempt to upload the model, retrying if necessary
        for attempt in range(self.max_retries):
            try:
                # Attempt to upload the model
                upload_id = self.api.model_upload(file_path=file_path, model_id=model_id, data_version=data_version, docker_image=docker_image)
                print(f"{api_type} model upload of '{file_path}' for '{model_name}' is successful! Upload ID: {upload_id}")
                return upload_id  # Return upload ID if successful
            except Exception as e:
                # Handle failed upload attempts
                if attempt < self.max_retries - 1:
                    print(f"Failed to upload model '{file_path}' for '{model_name}' to Numerai. Retrying in {self.sleep_time} seconds...")
                    print(f"Error: {e}")
                    time.sleep(self.sleep_time)  # Wait before retrying
                else:
                    # Handle final failed attempt
                    if self.fail_silently:
                        print(f"Failed to upload model '{file_path}' for '{model_name}' to Numerai. Skipping...")
                        print(f"Error: {e}")
                    else:
                        print(f"Failed to upload model '{file_path}' for '{model_name}' after {self.max_retries} attempts.")
                        raise e  # Raise the exception if not failing silently

    def get_available_data_versions(self) -> dict:
        """
        Retrieves the available data versions for model uploads.

        :return: A dictionary of available data versions.
        """
        # Call NumerAPI to get available data versions
        return self.api.model_upload_data_versions()

    def get_available_docker_images(self) -> dict:
        """
        Retrieves the available Docker images for model uploads.

        :return: A dictionary of available Docker images.
        """
        # Call NumerAPI to get available Docker images
        return self.api.model_upload_docker_images()

    def _get_model_id(self, model_name: str) -> str:
        """
        Retrieves the model ID for a given model name.

        :param model_name: The name of the model.
        :return: The ID of the model.

        Raises ValueError if the model name is not found in the user's Numerai account.
        """
        # Get the mapping of model names to model IDs
        model_mapping = self.get_model_mapping
        if model_name in model_mapping:
            return model_mapping[model_name]  # Return the model ID if found
        else:
            # Raise an error if the model name is not found
            available_models = ", ".join(model_mapping.keys())
            raise ValueError(f"Model name '{model_name}' not found in your Numerai account. " f"Available model names: {available_models}")

    @property
    def get_model_mapping(self) -> dict:
        """
        Retrieves the mapping of model names to their IDs from the user's Numerai account.

        :return: A dictionary mapping model names to model IDs.
        """
        # Call NumerAPI to get the model mapping
        return self.api.get_models()

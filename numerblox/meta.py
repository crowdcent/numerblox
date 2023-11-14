import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, _name_estimators, _final_estimator_has
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    check_memory,
    FLOAT_DTYPES,
)     
from sklearn.utils.metaestimators import available_if

from sklearn.utils import _print_elapsed_time


class MetaEstimator(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    """
    Helper for NumeraiPipeline and NumeraiFeatureUnion to use a model as a transformer.

    :param estimator: Underlying estimator like XGBoost, Catboost, scikit-learn, etc.
    :param predict_func: Name of the function that will be used for prediction.
    Must be one of 'predict', 'predict_proba', 'predict_log_proba'.
    For example, XGBRegressor has 'predict' and 'predict_proba' functions.
    :param model_type: "regressor" or "classifier". Used to determine if the estimator is multi output.
    """

    def __init__(self, estimator, predict_func="predict", model_type="regressor"):
        self.estimator = estimator
        if predict_func not in ["predict", "predict_proba", "predict_log_proba", "transform"]:
            raise ValueError("predict_func must be 'predict', 'predict_proba', 'predict_log_proba' or 'transform'.")
        self.predict_func = predict_func
        assert model_type in ["regressor", "classifier"], "model_type must be 'regressor' or 'classifier'."
        assert hasattr(self.estimator, self.predict_func), f"Estimator {self.estimator.__class__.__name__} does not have {self.predict_func} function."
        self.model_type = model_type
        # predict_proba for classifiers -> multi output
        self.proba_class_ = predict_func == "predict_proba" and model_type == "classifier"
        
    def fit(self, X: Union[np.array, pd.DataFrame], y, **kwargs):
        """
        Fit underlying estimator and set attributes.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True)
        # Either multi target or outputs are probabilities
        self.multi_output_ = len(y.shape) > 1 or self.proba_class_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)
        return self
    
    def transform(self, X: Union[np.array, pd.DataFrame], **kwargs) -> np.array:
        """
        Apply the `predict_func` on the fitted estimator.

        Shape `(X.shape[0], )` if estimator is not multi output and else `(X.shape[0], y.shape[1])`.
        All additional kwargs are passed to the underlying estimator's predict function.
        """
        check_is_fitted(self, "estimator_")
        output = getattr(self.estimator_, self.predict_func)(X, **kwargs)
        return output if self.multi_output_ else output.reshape(-1, 1)
    
    def predict(self, X: Union[np.array, pd.DataFrame], **kwargs) -> np.array:
        """ 
        For if a MetaEstimator happens to be the last step in the pipeline. Has same behavior as transform.
        """
        return self.transform(X, **kwargs)
    
    def get_feature_names_out(self, input_features = None) -> List[str]:
        check_is_fitted(self)
        feature_names = [f"{self.estimator.__class__.__name__}_{self.predict_func}_output"]
        return feature_names if not input_features else input_features


class CrossValEstimator(BaseEstimator, TransformerMixin):
    """
    Split your data into multiple folds and fit an estimator on each fold.
    For transforms predictions are concatenated into a 2D array.
    :param cv: Cross validation object that follows scikit-learn conventions.
    :param estimator: Estimator to fit on each fold.
    :param evaluation_func: Custom evaluation logic that is executed on validation data for each fold. Must accepts as input y_true and y_pred.
    For example, evaluation_func can handle logging metrics for each fold.
    Anything that evaluation_func returns is stored in `self.eval_results_`.
    :param predict_func: Name of the function that will be used for prediction.
    Must be one of 'predict', 'predict_proba', 'predict_log_proba'.
    For example, XGBRegressor has 'predict' and 'predict_proba' functions.
    :param verbose: Whether to print progress.
    """
    def __init__(self, estimator: BaseEstimator, cv: BaseCrossValidator, evaluation_func=None, predict_func="predict", verbose=False):
        super().__init__()
        self.cv = cv
        if not hasattr(self.cv, "split") or isinstance(self.cv, str):
            raise ValueError("cv must be a valid sklearn cv object withat least a 'split' function.")
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__name__
        self.evaluation_func = evaluation_func

        if predict_func not in ["predict", "predict_proba", "predict_log_proba"]:
            raise ValueError("predict_func must be 'predict', 'predict_proba', or 'predict_log_proba'.")
        self.predict_func = predict_func
        assert hasattr(self.estimator, self.predict_func), f"Estimator {self.estimator_name} does not have {self.predict_func} function."
        self.verbose = verbose

    def fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series], **kwargs):
        """ Use cross validation object to fit estimators. """
        self.estimators_ = []
        self.eval_results_ = []
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.reset_index(drop=True).values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.reset_index(drop=True).values
        for i, (train_idx, val_idx) in tqdm(enumerate(self.cv.split(X, y)), 
                                            desc=f"CrossValEstimator Fitting. Estimator='{self.estimator_name}'", 
                                            total=self.cv.get_n_splits(), 
                                            disable=not self.verbose):
            estimator = clone(self.estimator)
            if self.verbose:
                print(f"Fitting {self.estimator_name} on fold {len(self.estimators_)}")


            estimator.fit(X[train_idx], y[train_idx], **kwargs)

            # Execute custom evaluation logic
            if self.evaluation_func:
                if self.verbose:
                    print(f"Running evaluation on fold {len(self.estimators_)}")

                y_pred = getattr(estimator, self.predict_func)(X[val_idx])
                y_pred = self._postprocess_pred(y_pred)
                eval_fold = self.evaluation_func(y[val_idx], y_pred)
                if self.verbose:
                    print(f"CrossValEstimator (estimator='{self.estimator_name}'): Fold '{i}' evaluation results: '{eval_fold}'")
                self.eval_results_.append(eval_fold)

            self.estimators_.append(estimator)

            # Store output shape by doing inference on 1st sample of training set
            if i == 0:
                sample_prediction = getattr(estimator, self.predict_func)(X[train_idx][:1])
                sample_prediction = self._postprocess_pred(sample_prediction)
                self.output_shape_ = sample_prediction.shape[1:]
                self.multi_output_ = len(y.shape) > 1
                self.n_outputs_per_model_ = np.prod(self.output_shape_).astype(int)
        return self
    
    def transform(self, X, model_idxs: List[int] = None, **kwargs) -> np.array:
        """ 
        Use cross validation object to transform estimators. 
        :param X: Input data for inference.
        :param y: Target data for inference.
        :param model_idxs: List of indices of models to use for inference. 
        By default, all fitted models are used.
        :param kwargs: Additional arguments to pass to the estimator's predict function.
        """
        check_is_fitted(self)        
        inference_estimators = [self.estimators_[i] for i in model_idxs] if model_idxs else self.estimators_

        # Create an empty array to store predictions
        final_predictions = np.zeros((X.shape[0], len(inference_estimators) * self.n_outputs_per_model_))
        # Iterate through models to get predictions
        for idx, estimator in enumerate(inference_estimators):
            pred = getattr(estimator, self.predict_func)(X, **kwargs)
            pred = self._postprocess_pred(pred)
            
            # Calculate where to place these predictions in the final array
            start_idx = idx * self.n_outputs_per_model_
            end_idx = (idx + 1) * self.n_outputs_per_model_
            
            final_predictions[:, start_idx:end_idx] = pred

        return final_predictions

    def predict(self, X, model_idxs: List[int] = None, **kwargs) -> np.array:
        return self.transform(X, model_idxs, **kwargs)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        check_is_fitted(self)
        base_str = f"CrossValEstimator_{self.estimator_name}_{self.predict_func}"
        # Single-output case
        if self.n_outputs_per_model_ == 1:
            feature_names = [f"{base_str}_{i}" for i in range(len(self.estimators_))]
        # Multi-output case
        else:
            feature_names = []
            for i in range(len(self.estimators_)):
                for j in range(self.n_outputs_per_model_):
                    feature_names.append(f"{base_str}_{i}_output_{j}")
        return feature_names
    
    def _postprocess_pred(self, pred):
        # Make sure predictions are 2D
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)
        return pred
    
    def __sklearn_is_fitted__(self) -> bool:
        """ Check fitted status. """
        # Must have a fitted estimator for each split.
        return len(self.estimators_) == self.cv.get_n_splits()


class MetaPipeline(Pipeline):
    """
    Pipeline which turns all estimators into transformers by wrapping them in MetaEstimator.
    This allows to have pipeline steps after models.
    For example, a FeatureNeutralizer after an XGBRegressor.

    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    :param predict_func: Name of the function that will be used for prediction.
    """
    def __init__(self, steps, memory=None, verbose=False, predict_func="predict"):
        self.predict_func = predict_func
        self.modified_steps = self.wrap_estimators_as_transformers(steps)
        self.steps = self.modified_steps
        self.memory = memory
        self.verbose = verbose
    
    def wrap_estimators_as_transformers(self, steps):
        """
        Converts all estimator steps (except the last step) into transformers by wrapping them in MetaEstimator.
        :param steps: List of (name, transform) tuples specifying the pipeline steps.
        :return: Modified steps with all estimators wrapped as transformers.
        """
        transformed_steps = []
        for i, step_tuple in enumerate(steps):
            is_last_step = i == len(steps) - 1
            
            if len(step_tuple) == 3:
                name, step, columns = step_tuple
                transformed_steps.append(self._wrap_step(name, step, columns, is_last_step))
            else:
                name, step = step_tuple
                transformed_steps.append(self._wrap_step(name, step, is_last_step=is_last_step))
                
        return transformed_steps
    
    def _wrap_step(self, name, step, columns=None, is_last_step=False):
            """ Recursive function to wrap steps """
            # Recursive call
            if isinstance(step, (Pipeline, FeatureUnion, ColumnTransformer)):
                if isinstance(step, Pipeline):
                    transformed = step.__class__(self.wrap_estimators_as_transformers(step.steps))
                elif isinstance(step, FeatureUnion):
                    transformed = FeatureUnion(self.wrap_estimators_as_transformers(step.transformer_list))
                elif isinstance(step, ColumnTransformer):
                    transformed_transformers = self.wrap_estimators_as_transformers(step.transformers)
                    transformed = ColumnTransformer(transformed_transformers)
                return (name, transformed, columns) if columns else (name, transformed)

            # If it's the last step and it doesn't have a transform method, don't wrap it
            if is_last_step and not hasattr(step, 'transform'):
                return (name, step, columns) if columns else (name, step)

            # Wrap estimator that has the predict function but not the transform function
            elif hasattr(step, self.predict_func) and not hasattr(step, 'transform'):
                return (name, MetaEstimator(step, predict_func=self.predict_func))

            return (name, step, columns) if columns else (name, step)
    
    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = _fit_transform_one(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X
    
    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """ Overrides predict method in Pipeline so it also parses arguments to transforms."""
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            # Check if predict params is needed in transform. If so parse along
            sig = inspect.signature(transform.transform)
            params_needed = {k: v for k, v in predict_params.items() if k in sig.parameters}
            # Use the needed parameters when calling transform
            Xt = transform.transform(Xt, **params_needed)
        return self.steps[-1][1].predict(Xt, **predict_params)
    

def _fit_transform_one(
        transformer, X, y, weight, message_clsname="", message=None, **fit_params
    ):
        """
        Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
        with the fitted transformer. If ``weight`` is not ``None``, the result will
        be multiplied by ``weight``.
        """
        with _print_elapsed_time(message_clsname, message):
            sig = inspect.signature(transformer.transform)
            fit_params_needed = {k: v for k, v in fit_params.items() if k in sig.parameters}
            # Use the needed parameters when calling fit
            transformer.fit(X, y, **fit_params_needed)

            sig = inspect.signature(transformer.transform)
            transform_params_needed = {k: v for k, v in fit_params.items() if k in sig.parameters}
            # Use the needed parameters when calling transform
            res = transformer.transform(X, **transform_params_needed)

        if weight is None:
            return res, transformer
        return res * weight, transformer


def make_meta_pipeline(*steps, memory=None, verbose=False) -> MetaPipeline:
    """ 
    Convenience function for creating a MetaPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    """
    return MetaPipeline(_name_estimators(steps), memory=memory, verbose=verbose)

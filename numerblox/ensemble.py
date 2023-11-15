import scipy
import warnings
import numpy as np
import pandas as pd

from typing import Union, List
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
        

class NumeraiEnsemble(BaseEstimator, TransformerMixin):
    """
    Ensembler that standardizes predictions by era and averages them.
    :param weights: Sequence of weights (float or int), optional, default: None.
    If None, then uniform weights are used.
    :param n_jobs: The number of jobs to run in parallel for fit.
    Will revert to 1 CPU core if not defined.
    -1 means using all processors.
    :param donate_weighted: Whether to use Donate et al.'s weighted average formula.
    Often used when ensembling predictions from multiple folds over time.
    Paper Link: https://doi.org/10.1016/j.neucom.2012.02.053
    Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]
    """
    def __init__(self, weights=None, donate_weighted=False):
        super().__init__()
        self.weights = weights
        if self.weights and sum(self.weights) != 1:
            warnings.warn(f"Warning: Weights do not sum to 1. Got {sum(self.weights)}.")
        self.donate_weighted = donate_weighted

    def fit(self, X=None, y=None, **kwargs):
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[np.array, pd.DataFrame], eras: pd.Series) -> np.array:
        """ 
        Standardize by era and ensemble. 
        :param X: Input data where each column contains predictions from an estimator.
        :param eras: Era labels (strings) for each row in X.
        :return: Ensembled predictions.
        """
        check_is_fitted(self)
        assert len(X) == len(eras), f"input X and eras must have the same length. Got {len(X)} != {len(eras)}."

        if len(X.shape) == 1:
            raise ValueError("NumeraiEnsemble requires at least 2 prediction columns. Got 1.")
        
        n_models = X.shape[1]
        if n_models <= 1:
            raise ValueError(f"NumeraiEnsemble requires at least 2 predictions columns. Got {len(n_models)}.")
        
        # Override weights if donate_weighted is True
        if self.donate_weighted:
            weights = self._get_donate_weights(n=n_models)
        else:
            weights = self.weights
            
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Standardize predictions by era
        standardized_pred_list = []
        for i in range(n_models):
            # Skip standardization if all predictions are the same
            pred = X[:, i]
            if np.isnan(pred).any():
                warnings.warn(f"Warning: Some predictions in column '{i}' contain NaNs. Consider checking your estimators. Ensembled predictions will also be a NaN.")
            if np.all(pred == pred[0]):
                warnings.warn(f"Warning: Predictions in column '{i}' are all constant. Consider checking your estimators. Skipping these estimator predictions in ensembling.")
            else:
                standardized_pred = self._standardize_by_era(pred, eras)
                standardized_pred_list.append(standardized_pred)
        standardized_pred_arr = np.asarray(standardized_pred_list).T

        if not standardized_pred_list:
            raise ValueError("Predictions for all columns are constant. No valid predictions to ensemble.")

        # Average out predictions
        ensembled_predictions = np.average(standardized_pred_arr, axis=1, weights=weights)
        return ensembled_predictions.reshape(-1, 1)
    
    def predict(self, X: Union[np.array, pd.DataFrame], eras: pd.Series) -> np.array:
        """ 
        For if a NumeraiEnsemble happens to be the last step in the pipeline. Has same behavior as transform.
        """
        return self.transform(X, eras)

    def _standardize(self, X: np.array) -> np.array:
        """ 
        Standardize single era.
        :param X: Predictions for a single era.
        :return: Standardized predictions.
        """
        percentile_X = (scipy.stats.rankdata(X, method="ordinal") - 0.5) / len(X)
        return percentile_X
    
    def _standardize_by_era(self, X: np.array, eras: Union[np.array, pd.Series, pd.DataFrame]) -> np.array:
        """
        Standardize predictions of a single estimator by era.
        :param X: All predictions of a single estimator.
        :param eras: Era labels (strings) for each row in X.
        :return: Standardized predictions.
        """
        if isinstance(eras, (pd.Series, pd.DataFrame)):
            eras = eras.to_numpy().flatten()
        df = pd.DataFrame({'prediction': X, 'era': eras})
        df['standardized_prediction'] = df.groupby('era')['prediction'].transform(self._standardize)
        return df['standardized_prediction'].values.flatten()
    
    def _get_donate_weights(self, n: int) -> list:
        """
        Exponential weights as per Donate et al.'s formula.
        Example donate weighting for 3 folds: [0.25, 0.25, 0.5]
        Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]

        :param n: Number of estimators.
        :return: List of weights.
        """
        weights = []
        for j in range(1, n + 1):
            j = 2 if j == 1 else j
            weights.append(1 / (2 ** (n + 1 - j)))
        return weights

    def get_feature_names_out(self, input_features = None) -> List[str]:
        return ["numerai_ensemble_predictions"] if not input_features else input_features


class PredictionReducer(BaseEstimator, TransformerMixin):
    """ 
    Reduce multiclassification and proba preds to 1 column per model. 
    If predictions were generated with a regressor or regular predict you don't need this step.
    :param n_models: Number of resulting columns. 
    This indicates how many models were trained to generate the prediction array.
    :param n_classes: Number of classes for each prediction.
    If predictions were generated with predict_proba and binary classification -> n_classes = 2.
    """
    def __init__(self, n_models: int, n_classes: int):
        super().__init__()
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1. Got '{n_models}'.")
        self.n_models = n_models
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2. If n_classes = 1 you don't need PredictionReducer. Got '{n_classes}'.")
        self.n_classes = n_classes
        self.dot_array = [i for i in range(self.n_classes)]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: np.array):
        """
        :param X: Input predictions.
        :return: Reduced predictions of shape (X.shape[0], self.n_models).
        """
        check_is_fitted(self)
        reduced = []
        expected_n_cols = self.n_models * self.n_classes
        if len(X.shape) != 2:
            raise ValueError(f"Expected X to be a 2D array. Got '{len(X.shape)}' dimension(s).")
        if X.shape[1] != expected_n_cols:
            raise ValueError(f"Input X must have {expected_n_cols} columns. Got {X.shape[1]} columns while n_models={self.n_models} * n_classes={self.n_classes} = {expected_n_cols}. ")
        for i in range(self.n_models):
            # Extracting the predictions of the i-th model
            model_preds = X[:, i*self.n_classes:(i+1)*self.n_classes]
            r = model_preds @ self.dot_array
            reduced.append(r)
        reduced_arr = np.column_stack(reduced)
        return reduced_arr
    
    def predict(self, X: np.array):
        """ 
        For if PredictionReducer happens to be the last step in the pipeline. Has same behavior as transform.
        :param X: Input predictions.
        :return: Reduced predictions of shape (X.shape[0], self.n_models).
        """
        return self.transform(X)
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return [f"reduced_prediction_{i}" for i in range(self.n_models)] if not input_features else input_features
    
import scipy
import numpy as np
import pandas as pd

from typing import Union, List
from sklearn import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    FLOAT_DTYPES,
)


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
        self.donate_weighted = donate_weighted

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.array, pd.DataFrame], eras: pd.Series) -> np.array:
        """ 
        Standardize by era and ensemble. 
        :param X: Input data.
        :param eras: Era labels (strings) for each row in X.
        :return: Ensembled predictions.
        """
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
        # Standardize predictions by era
        standardized_pred_list = []
        for i in range(n_models):
            # Skip standardization if all predictions are the same
            pred = X[:, i]
            if np.all(pred == pred[0]):
                print("Warning: Some estimator predictions are constant. Consider checking your estimators. Skipping these estimator predictions in ensembling.")
            else:
                standardized_pred = self._standardize_by_era(pred, eras)
                standardized_pred_list.append(standardized_pred)
        standardized_pred_arr = np.asarray(standardized_pred_list).T

        # Average out predictions
        if not standardized_pred_list:
            raise ValueError("Predictions for all estimators are constant. No valid predictions to ensemble.")

        ensembled_predictions = np.average(standardized_pred_arr, axis=1, weights=weights)
        return ensembled_predictions
    
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
    
    def _standardize_by_era(self, X: np.array, eras: pd.Series) -> np.array:
        """
        Standardize predictions of a single estimator by era.
        :param X: All predictions of a single estimator.
        :param eras: Era labels (strings) for each row in X.
        :return: Standardized predictions.
        """
        df = pd.DataFrame({'prediction': X, 'era': eras})
        df['standardized_prediction'] = df.groupby('era')['prediction'].transform(self._standardize)
        return df['standardized_prediction'].values
    
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
    
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
        

class MetaEstimator(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    """
    Helper for NumeraiPipeline and NumeraiFeatureUnion to use a model as a transformer.

    :param estimator: Underlying estimator like XGBoost, Catboost, scikit-learn, etc.
    :param predict_func: Name of the function that will be used for prediction.
    Must be one of 'predict', 'predict_proba', 'predict_log_proba'.
    For example, XGBRegressor has 'predict' and 'predict_proba' functions.
    The estimator should have the function you define.
    """

    def __init__(self, estimator, predict_func="predict"):
        self.estimator = estimator
        assert predict_func in ["predict", "predict_proba", "predict_log_proba"], "predict_func must be 'predict', 'predict_proba' or 'predict_log_proba'."
        self.predict_func = predict_func
        assert hasattr(self.estimator, self.predict_func), f"Estimator {self.estimator.__class__.__name__} does not have {self.predict_func} function."
        
    def fit(self, X, y, **kwargs):
        """
        Fit underlying estimator and set attributes.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True)
        self.multi_output_ = len(y.shape) > 1
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
        return [f"{self.estimator.__class__.__name__}_{self.predict_func}_predictions"] if not input_features else input_features

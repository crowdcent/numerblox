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

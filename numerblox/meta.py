
import scipy
import numpy as np
import pandas as pd
from scipy import sparse

from typing import Union, List
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.ensemble import VotingRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin



class NumeraiEnsemble(VotingRegressor):
    """
    Ensembler that standardizes predictions by era and averages them.
    :param estimators: List of (name, estimator) tuples.
    :param eras: Era labels (strings) for each row in X.
    :param weights: Sequence of weights (float or int), optional, default: None.
    If None, then uniform weights are used.
    :param n_jobs: The number of jobs to run in parallel for fit.
    Will revert to 1 CPU core if not defined.
    -1 means using all processors.
    :param gaussianize: Whether to gaussianize predictions before standardizing.
    :param donate_weighted: Whether to use Donate et al.'s weighted average formula.
    Often used when ensembling predictions from multiple folds over time.
    Paper Link: https://doi.org/10.1016/j.neucom.2012.02.053
    Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]
    :param verbose: Whether to print info about ensembling.
    """
    def __init__(self, estimators: list, eras: np.array,
                 weights=None, n_jobs=None, gaussianize=True, 
                 donate_weighted=False, verbose=False):
        super().__init__(estimators=estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)
        self.eras = eras
        self.gaussianize = gaussianize
        self.donate_weighted = donate_weighted
        # Override weights if donate_weighted is True
        if self.donate_weighted:
            weights = self._get_donate_weights()
        else:
            weights = self._weights_not_none
        self.weights = weights

    def predict(self, X: Union[np.array, pd.DataFrame]) -> np.array:
        """ 
        Standardize by era and average. 
        :param X: Input data.
        :return: Ensembled predictions.
        """
        assert len(X) == len(self.eras), "input X and eras must have the same length."
        check_is_fitted(self)
        # Get raw predictions for each estimator
        pred_list = [est.predict(X) for est in self.estimators_]
        # Standardization by era
        standardized_pred_list = []
        for pred in pred_list:
            # Skip standardization if all predictions are the same
            if np.all(pred == pred[0]):
                print("Warning: Some estimator predictions are constant. Consider checking your estimators. Skipping these estimator predictions in ensembling.")
            else:
                standardized_pred = self._standardize_by_era(pred)
                standardized_pred_list.append(standardized_pred)

        if len(standardized_pred_list) == 0:
            raise ValueError("Predictions for all estimators are constant. No valid predictions to ensemble.")
        standardized_pred_arr = np.asarray(standardized_pred_list).T

        # Average out predictions
        ensembled_predictions = np.average(standardized_pred_arr, axis=1, weights=self.weights)

        # Raise error if values outside the [0...1] range.
        if not np.all((0 <= ensembled_predictions) & (ensembled_predictions <= 1)):
            raise ValueError("Ensembled predictions are not between 0 and 1. Consider checking your estimators.")
        
        return ensembled_predictions

    def _standardize(self, X: np.array) -> np.array:
        """ 
        Standardize single era.
        :param X: Predictions for a single era.
        :return: Standardized predictions.
        """
        percentile_X = (scipy.stats.rankdata(X, method="ordinal") - 0.5) / len(X)
        if self.gaussianize:
            percentile_X = scipy.stats.norm.ppf(percentile_X)
        return percentile_X
    
    def _standardize_by_era(self, X: np.array) -> np.array:
        """
        Standardize predictions of a single estimator by era.
        :param X: All predictions of a single estimator.
        :param eras: Era labels (strings) for each row in X.
        :return: Standardized predictions.
        """
        df = pd.DataFrame({'prediction': X, 'era': self.eras})
        df['standardized_prediction'] = df.groupby('era')['prediction'].transform(self._standardize)
        return df['standardized_prediction'].values
    
    def _get_donate_weights(self) -> list:
        """
        Exponential weights as per Donate et al.'s formula.
        Example donate weighting for 3 folds: [0.25, 0.25, 0.5]
        Example donate weighting for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]
        """
        weights = []
        for j in range(1, len(self.estimators) + 1):
            j = 2 if j == 1 else j
            weights.append(1 / (2 ** (len(self.estimators) + 1 - j)))
        return weights
    

class MetaEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    
    def transform(self, X, **predict_params):
        return self.estimator.predict(X, **predict_params)
    
    def predict(self, X, **predict_params):
        check_is_fitted(self)
        return self.estimator_.predict(X, **predict_params)
    

# TODO Create metaestimator that inherits from VotingRegressor and does (weighted) averaging of predictions
import scipy
import numpy as np
import pandas as pd
from typing import Union
from sklearn.ensemble import VotingRegressor
from sklearn.utils.validation import check_is_fitted


# TODO Add groupby for multiple eras in standardization.
class NumeraiEnsemble(VotingRegressor):
    def __init__(self, estimators, weights=None, n_jobs=None, gaussianize=True, verbose=False):
        super().__init__(estimators=estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)
        self.gaussianize = gaussianize

    def predict(self, X: Union[np.array, pd.DataFrame]) -> np.array:
        """ Normalize and average """
        check_is_fitted(self)
        # Get raw predictions
        pred_list = [est.predict(X) for est in self.estimators_]
        # Standardize each array
        standardized_pred_list = [self.standardize(x) for x in pred_list]
        # Combine
        standardized_pred_arr = np.asarray(standardized_pred_list).T
        # Average
        return np.average(standardized_pred_arr, axis=1, weights=self._weights_not_none)

    def standardize(self, X: np.array) -> np.array:
        X = (scipy.stats.rankdata(X, method="ordinal") - 0.5) / len(X)
        if self.gaussianize:
            X = scipy.stats.norm.ppf(X)
        return X



# class DonateWeightedEnsembler(BasePostProcessor):
#     """
#     Weighted average as per Donate et al.'s formula
#     Paper Link: https://doi.org/10.1016/j.neucom.2012.02.053
#     Code source: https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp

#     Weightings for 5 folds: [0.0625, 0.0625, 0.125, 0.25, 0.5]

#     :param cols: Prediction columns to ensemble.
#     Uses all prediction columns by default. \n
#     :param final_col_name: New column name for ensembled values.
#     :param verbose: Whether to print info about ensembling.
#     """
#     def __init__(self, final_col_name: str, cols: list = None, verbose: bool = True):
#         super().__init__(final_col_name=final_col_name, verbose=verbose)
#         self.cols = cols
#         self.n_cols = len(cols)
#         self.weights = self._get_weights()

#     def transform(self, X: NumerFrame, y=None) -> NumerFrame:
#         cols = self.cols if self.cols else X.prediction_cols
#         X.loc[:, self.final_col_name] = np.average(
#             X.loc[:, cols], weights=self.weights, axis=1
#         )
#         self._verbose_print_ensemble(self.cols)
#         return NumerFrame(X)

#     def _get_weights(self) -> list:
#         """Exponential weights."""
#         weights = []
#         for j in range(1, self.n_cols + 1):
#             j = 2 if j == 1 else j
#             weights.append(1 / (2 ** (self.n_cols + 1 - j)))
#         return weights



import inspect
import numpy as np
from scipy import sparse
import warnings
from sklearn.pipeline import Pipeline, FeatureUnion, _name_estimators

from numerblox.meta import MetaEstimator


class NumeraiPipeline(Pipeline):
    """
    Pipeline that allows for a neutralizer as the last step.
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    """
    def __init__(self, steps, memory=None, verbose=False):
        # Wrap model into a MetaEstimator so a neutralizer can come after it.
        if len(steps) >= 2 and not isinstance(steps[-2][1], MetaEstimator):
            steps[-2] = (steps[-2][0], MetaEstimator(steps[-2][1]))

        # Make sure the last step requires features and eras arguments
        if not self._has_required_args(steps[-1][1], "features", "eras"):
            warnings.warn(f"""NumeraiPipeline is mostly used for use cases where the last arguments are 'features' and 'eras'. For example, FeatureNeutralizer. Got '{steps[-1][1].__class__.__name__}'. Be sure to pass the right arguments into the .predict method and consider if a regular sklearn.pipeline.Pipeline also works.""")

        self.steps = steps
        self.memory = memory
        self.verbose = verbose

    def predict(self, X, **params):
        """Custom predict to handle additional arguments."""
        
        Xt = X
        for _, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        
        # Explicitly pass `features` and `eras` to the last step
        return self.steps[-1][-1].predict(Xt, **params)
    
    def transform(self, X, **params):
        """ Handle transform with predict. """
        return self.predict(X, **params)
    
    @staticmethod
    def _has_required_args(transformer, *args):
        sig = inspect.signature(transformer.predict)
        params = sig.parameters
        return all(arg in params for arg in args)
    

class NumeraiFeatureUnion(FeatureUnion):
    def transform(self, X, **params) -> np.array:
        # Apply transformer-specific transform parameters
        Xs = []
        for name, trans in self.transformer_list:
            if hasattr(trans, "predict"):
                if name in params:
                    result = trans.predict(X, **params[name])
                else:
                    result = trans.predict(X)
            elif hasattr(trans, "transform"):
                if name in params:
                    result = trans.transform(X, **params[name])
                else:
                    result = trans.transform(X)

            # If output is 1D, reshape to 2D array
            if len(result.shape) == 1:
                result = result.reshape(-1, 1)
            Xs.append(result)
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

def make_numerai_pipeline(*steps, memory=None, verbose=False) -> NumeraiPipeline:
    """ 
    Convenience function for creating a NumeraiPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    """
    return NumeraiPipeline(_name_estimators(steps), memory=memory, verbose=verbose)

def make_numerai_union(*transformers, n_jobs=None, verbose=False) -> NumeraiFeatureUnion:
    """
    Convenience function for creating a NumeraiFeatureUnion.
    :param transformers: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained.
    :param n_jobs: The number of jobs to run in parallel
    for fit. None means 1 unless in a joblib.parallel_backend context.
    -1 means using all processors.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    """
    return NumeraiFeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
        
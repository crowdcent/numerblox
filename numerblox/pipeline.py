
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, _name_estimators

from numerblox.meta import MetaEstimator


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
        self.steps = self.wrap_estimators_as_transformers(steps)
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


def make_meta_pipeline(*steps, memory=None, verbose=False) -> MetaPipeline:
    """ 
    Convenience function for creating a MetaPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    """
    return MetaPipeline(_name_estimators(steps), memory=memory, verbose=verbose)
        
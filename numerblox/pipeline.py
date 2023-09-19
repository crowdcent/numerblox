
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
        Converts all estimator steps into transformers by wrapping them in MetaEstimator.
        :param steps: List of (name, transform) tuples specifying the pipeline steps.
        :return: Modified steps with all estimators wrapped as transformers.
        """
        transformed_steps = []
        for i, step_tuple in enumerate(steps):
            if len(step_tuple) == 3:  # This is a ColumnTransformer step
                name, step, _ = step_tuple
            else:  # Standard 2-tuple (name, step)
                name, step = step_tuple
            # Handle nested Pipelines, FeatureUnions, and ColumnTransformers
            if isinstance(step, Pipeline):
                transformed_steps.append(
                    (name, step.__class__(self.wrap_estimators_as_transformers(step.steps)))
                )
            elif isinstance(step, FeatureUnion):
                transformed_steps.append(
                    (name, FeatureUnion(self.wrap_estimators_as_transformers(step.transformer_list)))
                )
            elif isinstance(step, ColumnTransformer):
                wrapped_transformers = self.wrap_estimators_as_transformers(step.transformers)
                # Since wrapped_transformers will now be in the format [(name, transformer, columns), ...]
                # you can directly use it to instantiate the new ColumnTransformer.
                transformed_steps.append((name, ColumnTransformer(wrapped_transformers)))

            # For the last step of any structure (main or nested), if it's not supposed to be a transformer
            elif i == len(steps) - 1 and not hasattr(step, 'transform'):
                transformed_steps.append((name, step))
            
            # Wrap if the step has a prediction method but no transform method
            elif hasattr(step, self.predict_func) and not hasattr(step, 'transform'):
                transformed_steps.append((name, MetaEstimator(step, predict_func=self.predict_func)))
            
            else:
                transformed_steps.append((name, step))
                
        return transformed_steps


def make_meta_pipeline(*steps, memory=None, verbose=False) -> MetaPipeline:
    """ 
    Convenience function for creating a MetaPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an instance of BaseNeutralizer.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    """
    return MetaPipeline(_name_estimators(steps), memory=memory, verbose=verbose)
        
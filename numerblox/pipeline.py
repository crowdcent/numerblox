from sklearn.base import Pipeline


class NumeraiPipeline(Pipeline):
    """
    :param verbose: Whether to print additional information during pipeline execution.
    """
    def __init__(self,
                 verbose=False):
        self.verbose = verbose
        
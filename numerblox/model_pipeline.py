import uuid
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Union, Dict

from .numerframe import NumerFrame
from .preprocessing import BasePreProcessor, CopyPreProcessor
from .model import BaseModel
from .postprocessing import Standardizer


class ModelPipeline:
    """
    Execute all preprocessing, prediction and postprocessing for a given setup.

    :param models: Initiliazed numerai-blocks Models (Objects inheriting from BaseModel) \n
    :param preprocessors: List of initialized Preprocessors. \n
    :param postprocessors: List of initialized Postprocessors. \n
    :param copy_first: Whether to copy the NumerFrame as a first preprocessing step. \n
    Highly recommended in order to avoid surprise behaviour by manipulating the original dataset. \n
    :param pipeline_name: Unique name for pipeline. Only used for display purposes.
    :param verbose: Whether to print additional information during pipeline execution.
    """
    def __init__(self,
                 models: List[BaseModel],
                 preprocessors: List[BasePreProcessor] = [],
                 postprocessors: List[BasePreProcessor] = [],
                 copy_first = True,
                 standardize = True,
                 pipeline_name: str = None, 
                 verbose=False):
        self.pipeline_name = pipeline_name if pipeline_name else uuid.uuid4().hex
        self.models = models
        self.copy_first = copy_first
        self.standardize = standardize
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors
        self.verbose = verbose

    def preprocess(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """ Run all preprocessing steps. Copies input by default. """
        if self.copy_first:
            dataf = CopyPreProcessor()(dataf)
        for preprocessor in tqdm(self.preprocessors,
                                 desc=f"{self.pipeline_name} Preprocessing:",
                                 position=0):
            if self.verbose:
                print(f":construction: Applying preprocessing: '[bold]{preprocessor.__class__.__name__}[/bold]' :construction:")
            dataf = preprocessor(dataf)
        return NumerFrame(dataf)

    def postprocess(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """ Run all postprocessing steps. Standardizes model prediction by default. """
        if self.standardize:
            dataf = Standardizer()(dataf)
        for postprocessor in tqdm(self.postprocessors,
                                  desc=f"{self.pipeline_name} Postprocessing: ",
                                  position=0):
            if self.verbose:
                print(f":construction: Applying postprocessing: '[bold]{postprocessor.__class__.__name__}[/bold]' :construction:")
            dataf = postprocessor(dataf)
        return NumerFrame(dataf)

    def process_models(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """ Run all models. """
        for model in tqdm(self.models,
                                  desc=f"{self.pipeline_name} Model prediction: ",
                                  position=0):
            if self.verbose:
                print(f":robot: Generating model predictions with '[bold]{model.__class__.__name__}[/bold]'. :robot:")
            dataf = model(dataf)
        return NumerFrame(dataf)

    def pipeline(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        """ Process full pipeline and return resulting NumerFrame. """
        preprocessed_dataf = self.preprocess(dataf)
        prediction_dataf = self.process_models(preprocessed_dataf)
        processed_prediction_dataf = self.postprocess(prediction_dataf)
        if self.verbose:
            print(f":checkered_flag: [green]Finished pipeline:[green] [bold blue]'{self.pipeline_name}'[bold blue]! :checkered_flag:")
        return processed_prediction_dataf

    def __call__(self, dataf: Union[pd.DataFrame, NumerFrame]) -> NumerFrame:
        return self.pipeline(dataf)


class ModelPipelineCollection:
    """
    Execute multiple initialized ModelPipelines in a sequence.

    :param pipelines: List of initialized ModelPipelines.
    """
    def __init__(self, pipelines: List[ModelPipeline]):
        self.pipelines = {pipe.pipeline_name: pipe for pipe in pipelines}
        self.pipeline_names = list(self.pipelines.keys())

    def process_all_pipelines(self, dataf: Union[pd.DataFrame, NumerFrame]) -> Dict[str, NumerFrame]:
        """ Process all pipelines and return Dictionary mapping pipeline names to resulting NumerFrames. """
        result_datafs = dict()
        for name in tqdm(self.pipeline_names,
                         desc="Processing Pipeline Collection"):
            result_datafs[name] = self.process_single_pipeline(dataf, name)
        return result_datafs

    def process_single_pipeline(self, dataf: Union[pd.DataFrame, NumerFrame], pipeline_name: str) -> NumerFrame:
        """ Run full model pipeline for given name in collection. """
        if self.verbose:
            print(f":construction_worker: [bold green]Processing model pipeline:[/bold green] '{pipeline_name}' :construction_worker:")
        pipeline = self.get_pipeline(pipeline_name)
        dataf = pipeline(dataf)
        return NumerFrame(dataf)

    def get_pipeline(self, pipeline_name: str) -> ModelPipeline:
        """ Retrieve model pipeline for given name. """
        available_pipelines = self.pipeline_names
        assert pipeline_name in available_pipelines, f"Requested pipeline '{pipeline_name}', but only the following models are in the collection: '{available_pipelines}'."
        return self.pipelines[pipeline_name]

    def __call__(self, dataf: Union[pd.DataFrame, NumerFrame]) -> Dict[str, NumerFrame]:
        return self.process_all_pipelines(dataf=dataf)

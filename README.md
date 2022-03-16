# NumerBlox
> Solid Numerai pipelines


`numerblox` offers Numerai specific functionality, so you can worry less about software/data engineering and focus more on building great Numerai models!

Most of the components in this library are designed for solid weekly inference pipelines, but tools like `NumerFrame`, preprocessors and evaluators also greatly simplify the training process.

**Questions and discussion:** [rocketchat.numer.ai/channel/numerblox](https://rocketchat.numer.ai/channel/numerblox)

**Documentation:** [crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox/)

![](https://img.shields.io/pypi/v/numerblox) ![](https://img.shields.io/pypi/pyversions/numerblox) ![](https://img.shields.io/github/contributors/crowdcent/numerblox) ![](https://img.shields.io/github/issues-raw/crowdcent/numerblox) ![](https://img.shields.io/github/repo-size/crowdcent/numerblox)

## 1. Install

`pip install numerblox`

## 2. How to use

### 2.1. Contents

#### 2.1.1. Core functionality

`numerblox` features the following functionality:

1. Downloading data (`NumeraiClassicDownloader` and `KaggleDownloader`)
2. A custom data structure extending Pandas DataFrame (`NumerFrame`)
3. A suite of preprocessors for Numerai Classic and Signals (feature selection, engineering and manipulation)
4. Model objects for easy inference.
5. A suite of postprocessors for Numerai Classic and Signals (standardization, ensembling, neutralization and penalization)
6. Pipelines handling processing and prediction (`ModelPipeline` and `ModelPipelineCollection`)
7. Evaluation (`NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator`)
8. Authentication (`Key` and `load_key_from_json`)
9. Submitting (`NumeraiClassicSubmitter` and `NumeraiSignalsSubmitter`)
10. Automated staking (`NumeraiClassicStaker` and `NumeraiSignalsStaker`)

#### 2.1.2. Educational notebooks

Example notebooks can be found in the `nbs/edu_nbs` directory.

`nbs/edu_nbs` currently contains the following examples:
- `numerframe_tutorial.ipynb`: A deep dive into what `NumerFrame` has to offer.
- `pipeline_construction.ipynb`: How to use `numerblox` tools for efficient Numerai inference.
- `submitting.ipynb`: How to use Submitters for safe and easy Numerai submissions.
- `google_cloud_storage.ipynb`: How to use Downloaders and Submitters to interact with Google Cloud Storage (GCS).
- `load_model_from_wandb.ipynb`: For [Weights & Biases](https://wandb.ai/) users. Easily pull a model from W&B for inference.

Development notebooks are also in the `nbs` directory. These notebooks are also used to generate the documentation.

**Questions or idea discussion for educational notebooks:** [rocketchat.numer.ai/channel/numerblox](https://rocketchat.numer.ai/channel/numerblox)

**Full documentation:** [crowdcent.github.io/numerblox](https://crowdcent.github.io/numerblox/)

### 2.2. Examples

Below we will illustrate a common use case for inference pipelines. To learn more in-depth about the features of this library, check out notebooks in `nbs/edu_nbs`.

#### 2.2.1. Numerai Classic

```python
# --- 0. Numerblox dependencies ---
from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe
from numerblox.postprocessing import FeatureNeutralizer
from numerblox.model import SingleModel
from numerblox.model_pipeline import ModelPipeline
from numerblox.key import load_key_from_json
from numerblox.submission import NumeraiClassicSubmitter

# --- 1. Download version 2 data ---
downloader = NumeraiClassicDownloader("data")
downloader.download_inference_data("current_round")

# --- 2. Initialize NumerFrame ---
metadata = {"version": 2,
            "joblib_model_name": "test",
            "joblib_model_path": "test_assets/joblib_v2_example_model.joblib",
            "numerai_model_name": "test_model1",
            "key_path": "test_assets/test_credentials.json"}
dataf = create_numerframe(file_path="data/current_round/numerai_tournament_data.parquet",
                          metadata=metadata)

# --- 3. Define and run pipeline ---
models = [SingleModel(dataf.meta.joblib_model_path,
                      model_name=dataf.meta.joblib_model_name)]
# No preprocessing and 0.5 feature neutralization
postprocessors = [FeatureNeutralizer(pred_name=f"prediction_{dataf.meta.joblib_model_name}",
                                     proportion=0.5)]
pipeline = ModelPipeline(preprocessors=[],
                         models=models,
                         postprocessors=postprocessors)
dataf = pipeline(dataf)

# --- 4. Submit ---
# Load credentials from .json (random credentials in this example)
key = load_key_from_json(dataf.meta.key_path)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# full_submission checks contents, saves as csv and submits.
submitter.full_submission(dataf=dataf,
                          cols=f"prediction_{dataf.meta.joblib_model_name}_neutralized_0.5",
                          model_name=dataf.meta.numerai_model_name,
                          version=dataf.meta.version)

# --- 5. Clean up environment (optional) ---
downloader.remove_base_directory()
submitter.remove_base_directory()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Directory structure before starting                                                              
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 test_assets                                                                                  
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 joblib_v2_example_model.joblib                                                           
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 test_credentials.json                                                                    
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Directory structure after submitting                                                             
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 data                                                                                         
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📁 current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">┃       ┗━━ </span>📄 numerai_tournament_data.parquet                                                      
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 sub_current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 test_model1.csv                                                                          
</pre>



#### 2.2.2. Numerai Signals

```python
# --- 0. Numerblox dependencies ---
from numerblox.download import KaggleDownloader
from numerblox.numerframe import create_numerframe
from numerblox.preprocessing import KatsuFeatureGenerator
from numerblox.model import SingleModel
from numerblox.model_pipeline import ModelPipeline
from numerblox.key import load_key_from_json
from numerblox.submission import NumeraiSignalsSubmitter

# --- 1. Download Katsu1110 yfinance dataset from Kaggle ---
kd = KaggleDownloader("data")
kd.download_inference_data("code1110/yfinance-stock-price-data-for-numerai-signals")

# --- 2. Initialize NumerFrame with metadata ---
metadata = {"numerai_model_name": "test_model1",
            "key_path": "test_assets/test_credentials.json"}
dataf = create_numerframe("data/full_data.parquet", metadata=metadata)

# --- 3. Define and run pipeline ---
models = [SingleModel("models/signals_model.cbm", model_name="cb")]
# Simple and fast feature generator based on Katsu Signals starter notebook
# https://www.kaggle.com/code1110/numeraisignals-starter-for-beginners
pipeline = ModelPipeline(preprocessors=[KatsuFeatureGenerator(windows=[20, 40, 60])],
                         models=models,
                         postprocessors=[])
dataf = pipeline(dataf)

# --- 4. Submit ---
# Load credentials from .json (random credentials in this example)
key = load_key_from_json(dataf.meta.key_path)
submitter = NumeraiSignalsSubmitter(directory_path="sub_current_round", key=key)
# full_submission checks contents, saves as csv and submits.
# cols selection must at least contain 1 ticker column and a signal column.
dataf['signal'] = dataf['prediction_cb']
submitter.full_submission(dataf=dataf,
                          cols=['bloomberg_ticker', 'signal'],
                          model_name=dataf.meta.numerai_model_name)

# --- 5. Clean up environment (optional) ---
kd.remove_base_directory()
submitter.remove_base_directory()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Directory structure before starting                                                              
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 test_assets                                                                                  
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📄 test_credentials.json                                                                    
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 models                                                                                       
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 signals_model.cbm                                                                        
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Directory structure after submitting                                                             
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 data                                                                                         
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📄 full_data.parquet                                                                        
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 sub_current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 submission.csv                                                                           
</pre>



## 3. Contributing

Be sure to read `CONTRIBUTING.md` for detailed instructions on contributing.

If you have questions or want to discuss new ideas for `numerblox`, check out [rocketchat.numer.ai/channel/numerblox](https://rocketchat.numer.ai/channel/numerblox).



## 4. Branch structure


Every new feature should be implemented in a branch that branches from `dev` and has the naming convention `feature/{FEATURE_DESCRIPTION}`. Explicit bugfixes should be named `bugfix/{FIX_DESCRIPTION}`. An example structure is given below.


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Branch structure                                                                                    
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📦 main (release)                                                                               
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>👨‍💻 dev                                                                                    
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/ta-signals-features                                                          
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/news-api-downloader                                                          
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/staking-portfolio-management                                                 
<span style="color: #808080; text-decoration-color: #808080">        ┗━━ </span>✨ bugfix/evaluator-metrics-fix                                                         
</pre>




## 5. Crediting sources

Some of the components in this library may be based on forum posts, notebooks or ideas made public by the Numerai community. We have done our best to ask all parties who posted a specific piece of code for their permission and credit their work in the documentation. If your code is used in this library without credits, please let us know, so we can add a link to your article/code.

If you are contributing to `numerblox` and are using ideas posted earlier by someone else, make sure to credit them by posting a link to their article/code in documentation.

<img src="nbs/assets/images/crowdcent_logo.png" width="300" height="300" style="max-width: 300px">

**- CrowdCent**

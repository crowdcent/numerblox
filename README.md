# NumerBlox
> Tools for solid Numerai pipelines


## 1. Install

`pip install numerblox`

## 2. How to use

### 2.1. Contents

Example and educational notebooks can be found in the `edu_nbs` directory. Development notebooks are in the `nbs` directory.

The library features the following tools to build your Numerai pipelines:

1. `download`
2. `numerframe`
3. `preprocessing`
4. `model`
5. `postprocessing`
6. `ModelPipeline` (and `ModelPipelineCollection`)
7. `evaluation`
8. `Key` (containing authentication info)
9. `NumeraiClassicSubmitter` and `NumeraiSignalsSubmitter`
10. `staking`

### 2.2. Examples

Below we will illustrate a common use case for inference pipelines. To learn more in-depth about the features of this library, check out notebooks in the `edu_nbs` directory.

#### 2.2.1. Numerai Classic

```
#other
#hide_output

# --- 1. Download version 2 data ---
downloader = NumeraiClassicDownloader("data")
downloader.download_inference_data("current_round")

# --- 2. Initialize NumerFrame ---
metadata = {"version": 2,
            "joblib_model_name": "test",
            "joblib_model_path": "test_assets/joblib_v2_example_model.joblib",
            "numerai_model_name": "test_model1",
            "key_path": "test_assets/test_credentials.json"
            }
dataf = create_numerframe(file_path="data/current_round/numerai_tournament_data.parquet",
                          metadata=metadata)

# --- 3. Define and run pipeline ---
model1 = SingleModel(dataf.meta.joblib_model_path,
                     model_name=dataf.meta.joblib_model_name)
# No preprocessing and 0.5 feature neutralization
pipeline = ModelPipeline(preprocessors=[],
                         models=[model1],
                         postprocessors=[FeatureNeutralizer(
                             pred_name=f"prediction_{dataf.meta.joblib_model_name}",
                             proportion=0.5
                         )]
                         )
dataset = pipeline(dataf)

# --- 4. Submit ---
# Random credentials
key = load_key_from_json(dataf.meta.key_path)
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# Only works with valid key credentials
submitter.full_submission(dataf=dataf,
                          cols=f"prediction_{dataf.meta.joblib_model_name}_neutralized_0.5",
                          file_name=f"{dataf.meta.numerai_model_name}.csv",
                          model_name=dataf.meta.numerai_model_name,
                          version=dataf.meta.version
                          )

# --- 5. Clean up environment (optional) ---
downloader.remove_base_directory()
submitter.remove_base_directory()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Structure before starting                                                                        
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 test_assets                                                                                  
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 joblib_v2_example_model.joblib                                                           
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 test_credentials.json                                                                    
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Structure after submitting                                                                       
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 data                                                                                         
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📁 current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">┃       ┗━━ </span>📄 numerai_tournament_data.parquet                                                      
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 sub_current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 test_model1.csv                                                                          
</pre>



## 3. Contributing

### 3.1. Overview

Below are a few guidelines for development of `numerblox`. Also be sure to read `CONTRIBUTING.md` for more detailed instruction on contributing.

Thanks a lot for wanting to help us out with this project! We are using a project setup called [nbdev](https://nbdev.fast.ai/) to easily develop code, documentation and tests within Jupyter notebooks. If you are only using the library you don't have to worry about this. Just pip install and you are good to go!

If you are thinking of contributing and are not familiar with nbdev, it may take some time to learn nbdev development. We are happy to help out and point you to documentation or videos to learn more.

If you are interested in the full scope of what nbdev has to offer, check out this tutorial with Jeremy Howard:
 [https://youtu.be/Hrs7iEYmRmg](https://youtu.be/Hrs7iEYmRmg).

Why are we using nbdev? To learn more about the rationale behind nbdev:
[https://youtu.be/9Q6sLbz37gk](https://youtu.be/9Q6sLbz37gk)

nbdev live coding example with Hamel Husain:
[https://youtu.be/ZJTop5uqC2U](https://youtu.be/ZJTop5uqC2U)



### 3.2. Bugs / Issues / Enhancements.

Even though most of the components in this library are tested, the project is still in an early stage of development. If you discover bugs, other issues or ideas for enhancements, do not hesitate to make a Github issue. Describe in the issue what code was run on what machine and background on the issue. Add stacktraces and screenshots if this is relevant for solving the issue. Also, please define appropriate labels for the Github issue.

### 3.3. Contributing Code

There are a few small things you should do before contributing code to this project. After you clone the repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which cleans up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran). This avoids unnecessary merge conflicts.

Before pushing code to the branch you are working in, be sure to run `nbdev_build_lib` and `nbdev_build_docs` so all code is synced.



### 3.4. Branch structure


Every new feature should be implemented in a branch that branches from `dev` and has the naming convention `feature/{FEATURE_DESCRIPTION}`. Explicit bugfixes should be names `bugfix/{FIX_DESCRIPTION}`. An example structure is given below.


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Branch structure                                                                                    
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📦 main (release)                                                                               
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>👨‍💻 dev                                                                                    
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/ta-signals-features                                                          
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/news-api-downloader                                                          
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/staking-portfolio-management                                                 
<span style="color: #808080; text-decoration-color: #808080">        ┗━━ </span>✨ bugfix/evaluator-metrics-fix                                                         
</pre>



## 4. Crediting sources

Some of the components in this library may be based on forum posts, notebooks or ideas made public by the Numerai community. We have done our best to ask all parties who posted a specific piece of code for their permission and credit their work in the documentation. If your code is used in this library without credits, please let us know, so we can add a link to your article/code.

If you are contributing to `numerblox` and are using ideas posted earlier by someone else, make sure to credit them by posting a link to their article/code in documentation.

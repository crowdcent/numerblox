# Numerai Blocks
> Tools for solid Numerai pipelines


## 1. Install

`pip install numerai-blocks`

## 2. How to use

### 2.1. Contents

Example and educational notebooks can be found in the `edu_nbs` directory. Development notebooks are in the `nbs` directory.

The library features the following tools to build your Numerai pipelines:

1. Downloaders
2. NumerFrame
3. Preprocessing
4. Model
5. Postprocessing
6. ModelPipeline (and ModelPipelineCollection)
7. Evaluators
8. Key (containing authentication info)
9. Submittors
10. Staking functionality

### 2.2. Examples

Below we will illustrate a few base use cases for inference pipelines. To learn more in-depth about the features of the framework check out notebooks in the `edu_nbs` directory.

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
submittor = NumeraiClassicSubmittor(directory_path="sub_current_round", key=key)
# Only works with valid key credentials
submittor.full_submission(dataf=dataf,
                          cols=f"prediction_{dataf.meta.joblib_model_name}_neutralized_0.5",
                          file_name=f"{dataf.meta.numerai_model_name}.csv",
                          model_name=dataf.meta.numerai_model_name,
                          version=dataf.meta.version
                          )

# --- 5. Clean up environment (optional) ---
downloader.remove_base_directory()
submittor.remove_base_directory()
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




    <IPython.core.display.Javascript object>


## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

### Branch structure


Every new feature should be implemented a branch that branches from `dev` and has the naming convention `feature/{FEATURE_DESCRIPTION}`.


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Branch structure                                                                                    
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📦 main (release)                                                                               
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>👨‍💻 dev                                                                                    
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/1                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/2                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┗━━ </span>✨ feature/3                                                                            
</pre>




    <IPython.core.display.Javascript object>


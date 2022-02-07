# Numerai Blocks
> Tools for solid Numerai pipelines


## 1. Install

`pip install numerai-blocks`

## 2. How to use

### 2.1. Contents

Example and educational notebooks can be found in the `edu_nbs` directory. Development notebooks are in the `nbs` directory.

The library features the following tools to build your Numerai pipelines:

- Downloaders
- NumerFrame
- Preprocessing
- Model
- ModelPipeline and ModelPipelineCollection
- Postprocessing
- Evaluators
- Key (containing authentication info)
- Submittors
- Staking functionality

### 2.2. Quick Examples

#### 2.2.1. Numerai Classic

```
# slow
# Download version 2 data
# downloader = NumeraiClassicDownloader("data")
# downloader.download_inference_data("current_round")
#
# # Initialize Dataset
# metadata = {"version": 2, "model_name": "MY_MODEL"}
# dataset = create_numerframe(file_path="data/current_round/numerai_tournament_data.parquet", metadata=metadata)
#
# # Define and run pipeline
# model1 = JoblibModel(model_directory="dir_with_joblib_models",
#                      model_name="test_model")
# pipeline = ModelPipeline(pipeline_name=dataset.base_model_name,
#                              preprocessors=[],
#                              models=[model1],
#                              postprocessors=[FeatureNeutralizer(proportion=0.5)])
# dataset = pipeline(dataset)
#
# # Submit
# key = load_key_from_json("my_key.json")
# submittor = NumeraiClassicSubmittor(directory_path="sub_current_round", key=key)
# submittor.full_submission(dataf=dataset.dataf,
#                           cols="prediction_test_model_neutralized_0.5",
#                           file_name=f"{dataset.model_name}.csv",
#                           model_name=dataset.model_name,
#                           version=dataset.version
#                           )
#
# # Remove data and subs
# downloader.remove_base_directory()
# submittor.remove_base_directory()
```


    <IPython.core.display.Javascript object>



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Structure before starting                                                                        
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📄 my_key.json                                                                                  
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 dir_with_joblib_models                                                                       
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model1.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model2.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model3.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model4.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 model5.joblib                                                                            
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">💻 Structure after submitting                                                                       
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 data                                                                                         
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📁 current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">┃       ┗━━ </span>📄 numerai_tournament_data.parquet                                                      
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📁 sub_current_round                                                                            
<span style="color: #808080; text-decoration-color: #808080">┃   ┗━━ </span>📄 MY_MODEL.csv                                                                             
<span style="color: #808080; text-decoration-color: #808080">┣━━ </span>📄 my_key.json                                                                                  
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📁 dir_with_joblib_models                                                                       
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model1.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model2.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model3.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┣━━ </span>📄 model4.joblib                                                                            
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>📄 model5.joblib                                                                            
</pre>




    <IPython.core.display.Javascript object>


### 2.2.2. Numerai Signals

```
# slow
```


    <IPython.core.display.Javascript object>


## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

### Branch structure



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Branch structure                                                                                    
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📦 main (release)                                                                               
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>👨‍💻 dev                                                                                    
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/1                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/2                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┗━━ </span>✨ feature/3                                                                            
</pre>




    <IPython.core.display.Javascript object>


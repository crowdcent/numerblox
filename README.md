# Numerai Blocks
> Tools for solid Numerai pipelines


This file will become your README and also the index of your documentation.

## Install

`pip install numerai-blocks`

## How to use

Development notebooks are in the `dev_nbs` directory. Example and educational notebooks can be found in the `edu_nbs` directory.

1. Downloaders
2. Dataloaders
3. Dataset objects (with arbitrary metadata)
4. Preprocessors and Postprocessors
5. Model, ModelPipeline and ModelPipelineCollection (Or FeatureUnion??)
6. Predictions (with arbitrary metadata)
7. Evaluators
8. Submittors
9. Key (containing authentication info)

```
1+1
```




    2



```
import uuid
print(uuid.uuid4())
```

    649ce02a-6fb4-4b1f-8375-d3b46ed391c9


## Contributing

After you clone this repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

### Branch structure


```
console.print(tree)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Brranch structure                                                                                   
<span style="color: #808080; text-decoration-color: #808080">┗━━ </span>📦 main                                                                                         
<span style="color: #808080; text-decoration-color: #808080">    ┗━━ </span>👨‍💻 dev                                                                                    
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/1                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┣━━ </span>✨ feature/2                                                                            
<span style="color: #808080; text-decoration-color: #808080">        ┗━━ </span>✨ feature/3                                                                            
</pre>



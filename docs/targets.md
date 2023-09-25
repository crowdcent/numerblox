# Target Engineering

Target engineering object allows you to easily create synthetic targets to train on or to convert raw price data into Numerai-style targets.

## Why?

### Why use Target Processors?

- **Enhanced Experimentation**: The availability of synthetic targets through the `BayesianGMMTargetProcessor` allows modelers to test new algorithms, techniques, or strategies.

- **Align with Numerai's Methodology**: `SignalsTargetProcessor` ensures that the targets you use are consistent with Numerai's approach. This alignment boosts the relevance of your models, potentially leading to better performance in the competition.

- **Versatility**: With different windows and target types, `SignalsTargetProcessor` offers a rich set of features, allowing for a more nuanced approach to model training. By exploring different timeframes and target representations, you can gain a deeper understanding of the data's dynamics.

- **Efficiency**: Manually engineering features or creating synthetic targets can be time-consuming and error-prone. These processors automate intricate steps, saving time and ensuring accuracy.

By integrating these processors into your workflow, you can enhance your modeling capabilities, streamline experimentation, and align closer to Numerai's expectations.

## BayesianGMMTargetProcessor

The `BayesianGMMTargetProcessor`` generates synthetic targets based on a Bayesian Gaussian Mixture model. It's primarily used for creating fake targets, which are useful for experimenting and validating model structures without exposing true labels.

### Example:
```py
from numerblox.targets import BayesianGMMTargetProcessor
processor = BayesianGMMTargetProcessor(n_components=3)
processor.fit(X=train_features, y=train_targets, eras=train_eras)
fake_target = processor.transform(X=train_features, eras=train_eras)
```

For more detailed examples and use-cases, check out `examples/synthetic_data_generation.ipynb.`


## SignalsTargetProcessor

The `SignalsTargetProcessor` is specifically designed to engineer targets for Numerai Signals. This involves converting raw price data into Numerai-style targets.

### Example:
```py
from numerblox.target_processing import SignalsTargetProcessor
processor = SignalsTargetProcessor(price_col="close")
signals_target_data = processor.transform(dataf=data, eras=eras_column)
```
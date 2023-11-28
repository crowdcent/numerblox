# Evaluators

NumerBlox offers evaluators for both Numerai Classic and Numerai Signals.

## Common Metrics

The following metrics are included for `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator`:

For both `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator` you can set a custom `metrics_list` with all metrics you want to compute.

All valid metrics for `metrics_list` you can use are:

- "mean_std_sharpe" -> Mean, standard deviation and Sharpe ratio based on Corrv2 (Numerai Correlation).

- "apy" -> Annual Percentage Yield.

- "max_drawdown" -> Max drawdown.

- "calmar_ratio" -> [Calmar Ratio](https://www.investopedia.com/terms/c/calmarratio.asp).

- "autocorrelation" -> Autocorrelation (1st order).

- "max_feature_exposure" -> [Max feature exposure](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899).

- "smart_sharpe" -> Smart Sharpe.

- "legacy_mean_std_sharpe" -> Mean, standard deviation and Sharpe ratio based on legacy model contribution.

- "fn_mean_std_sharpe" -> [Feature Neutral](https://docs.numer.ai/tournament/feature-neutral-correlation) mean, standard deviation and Sharpe ratio (can take some time to compute).

- "tb200_mean_std_sharpe" -> Mean, standard deviation and Sharpe ratio based on TB200.

- "tb500_mean_std_sharpe" -> Mean, standard deviation and Sharpe ratio based on TB500.

The following metrics only work in `benchmark_cols` is defined in `full_evaluation`:
- "mc_mean_std_sharpe" -> Mean, standard deviation and Sharpe ratio based on model contribution.

- "corr_with" -> Correlation with benchmark predictions.

- "ex_diss" -> [Exposure Dissimilarity](https://forum.numer.ai/t/true-contribution-details/5128/4) to benchmark predictions.

By default, metrics will include `["mean_std_sharpe", "apy", "max_drawdown", "calmar_ratio"]`

## Numerai Classic specific metrics

`NumeraiClassicEvaluator` can also compute [FNCv3](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation#fnc-on-the-website). If you want to compute this add `fncv3_mean_std_sharpe` to the `metrics_list`.

```py
from numerblox.evaluation import NumeraiClassicEvaluator, FAST_METRICS

# Validation DataFrame to compute metrics on
# Should have at least era_col, pred_cols and target_col columns.
val_df = ...

evaluator = NumeraiClassicEvaluator(era_col="era", metrics_list=FAST_METRICS)
metrics = evaluator.full_evaluation(val_df, 
                                    pred_cols=["prediction"], 
                                    target_col="target",
                                    benchmark_cols=["benchmark1", "benchmark2"])
```

## Numerai Signals specific metrics

`NumeraiSignalsEvaluator` offers neutralized correlation scores. This is a special operation as it calls on Numerai's servers and needs additional authentication so it is not included in `full_evaluation`. It can still be beneficial to calculate as this metric is close to the one used for payouts.

Example of how to get neutralized correlation scores for Numerai Signals:
```py
from numerblox.misc import Key
from numerblox.evaluation import NumeraiSignalsEvaluator

evaluator = NumeraiSignalsEvaluator()

# A Numerai Signals model name you use.
model_name = "MY_MODEL"
# NumerBlox Key for accessing the Numerai API
key = Key(pub_id="Hello", secret_key="World")
# DataFrame with validation data containing prediction, friday_date, ticker and data_type columns
val_df = pd.DataFrame()

evaluator.get_neutralized_corr(val, model_name=model_name, key=key)
# Returns a Pandas Series like this.
# pd.Series([0.01, ..., 0.02])
```

## Custom functions

Evaluators can be augmented with custom metrics that will be executed in addition to the default metrics. This can be done by passing a list of functions to the `custom_functions` argument when initializing the evaluator. Custom functions work both in `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator`.

Each custom function should:
- Be a callable (function or class that implements \_\_call\_\_).
- Have the following function signature:
    - dataf: DataFrame passed into evaluation (pd.DataFrame).
    - pred_col: Column name containing the predictions to evaluate (str).
    - target_col: Column name with main target to evaluate against (str).

Example of how to use custom functions in `NumeraiClassicEvaluator`:
```py
from numerblox.evaluation import NumeraiClassicEvaluator

def my_custom_function(dataf: pd.DataFrame, pred_col: str, target_col: str) -> float:
    """ Dummy evaluation function. """
    return 0.5

evaluator = NumeraiClassicEvaluator(custom_functions=[my_custom_function])

# In evaluator my_custom_function(dataf=val_df, pred_col="prediction", target_col="target") is called.
metrics = evaluator.full_evaluation(val_df, 
                                    pred_cols=["prediction"], 
                                    target_col="target")
# metrics will contain a "my_custom_function" column with value 0.5.
```
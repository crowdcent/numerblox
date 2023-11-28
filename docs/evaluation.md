# Evaluators

NumerBlox offers evaluators for both Numerai Classic and Numerai Signals.

## Common Metrics

The following metrics are included for `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator`:

- Mean, Standard Deviation and Sharpe for era returns (Corrv2 aka Numerai Correlation).

- Smart Sharpe.

- Max drawdown.

- Annual Percentage Yield (APY).

- [Max feature exposure](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899).

- [Feature Neutral](https://docs.numer.ai/tournament/feature-neutral-correlation) Mean, Standard deviation and Sharpe.

- Autocorrelation (1st order).

- [Calmar Ratio](https://www.investopedia.com/terms/c/calmarratio.asp).

- Mean, Standard Deviation and Sharpe for TB200 and TB500 (Buy top 200/500 stocks and sell bottom 200/500 stocks).

- Performance vs. optional benchmark predictions.

- [Exposure Dissimilarity](https://forum.numer.ai/t/true-contribution-details/5128/4) to benchmark predictions.

- Model contribution (mc) and legacy model contribution (legacy_mc) against benchmark predictions. Model contribution calculations require defining `benchmark_cols` to compare against.

For both `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator` you can set `fast_mode=True` to skip max. feature exposure, [FNCV3 metrics](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation#fnc-on-the-website), TB200, TB500 and exposure dissimilarity. These metrics in particular can take a while to compute.

## Numerai Classic specific metrics

`NumeraiClassicEvaluator` will also compute [FNCv3](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation#fnc-on-the-website). The FNCV3 mean is a common metric shown on the Numerai leaderboard under `FNCV3`. `NumeraiClassicEvaluator` will compute the mean, standard deviation and Sharpe ratio for FNCV3. 

```py
from numerblox.evaluation import NumeraiClassicEvaluator

# Validation DataFrame to compute metrics on
# Should have at least era_col, pred_cols and target_col columns.
val_df = ...

evaluator = NumeraiClassicEvaluator(era_col="era")
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
- Have the following input arguments:
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
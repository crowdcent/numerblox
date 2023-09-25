# Evaluators

NumerBlox offers evaluators for both Numerai Classic and Numerai Signals.

## Common Metrics

The following metrics are included for both:
- Mean, Standard Deviation and Sharpe for era returns (Numerai Correlation).

- Max drawdown.

- Annual Percentage Yield (APY).

- Correlation with example predictions.

- [Max feature exposure](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899).

- [Feature Neutral](https://docs.numer.ai/tournament/feature-neutral-correlation) Mean, Standard deviation and Sharpe.

- [Exposure Dissimilarity](https://forum.numer.ai/t/true-contribution-details/5128/4).

- [Calmar Ratio](https://www.investopedia.com/terms/c/calmarratio.asp).

- Mean, Standard Deviation and Sharpe for TB200 (Buy top 200 stocks and sell bottom 200 stocks).

- Mean, Standard Deviation and Sharpe for TB500 (Buy top 500 stocks and sell bottom 500 stocks).

For both `NumeraiClassicEvaluator` and `NumeraiSignalsEvaluator` you can set `fast_mode=True` to skip max. features exposure, feature neutral mean, standard deviation and sharpe, TB200 and TB500 and exposure dissimilarity. These metrics in particular can take a while to compute.

## Numerai Classic specific metrics

`NumeraiClassicEvaluator` specifically will also compute the Feature Neutral Mean, Standard deviation and Sharpe based on the FNCV3 Features. The FNCV3 mean is a common metric shown on the Numerai leaderboard under `FNCV3`.

## Numerai Signals specific metrics

`NumeraiSignalsEvaluator` specifically offers neutralized correlation scores that are calculated on Numerai diagnostics. This is a special operation as it calls on Numerai's servers and needs additional authentication so it is not included in the regular `full_evaluation`.


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
# Submitters

NumerBlox provides submitters for both Numerai Classic and Signals. 
Also check out `example/submitting.ipynb` for more information on Numerai submission.

## Why?
- **Simplified Workflow**: Instead of managing multiple manual steps for submissions, `Submitters` allow you to simplify the submission process down to a few lines of code.

- **Integrated Validation Checks**: Before submitting, `Submitters` performs a series of checks to ensure the submission format is correct and prevent common mistakes that could lead to invalid submissions.

- **Security**: By providing a way to load credentials from a `.json` file, `Submitters` ensures that you're not hard-coding your secret credentials in the main code, reducing the risk of accidental exposure.

- **Automatic Cleanup**: For users who run automated jobs, the ability to automatically clean up the environment post-submission ensures that your workspace remains clutter-free.

With `Submitters`, you can focus more on developing and refining your model and spend less time on the manual aspects of the submission process.

## Instantiation

In order to use a Submitter you should first create a `Key` object which handles credentials.
There are two ways to create a `Key`:

**1. Initialize `Key` with `pub_id` and `secret_key` from memory.**

```py
from numerblox.misc import Key
key = Key(pub_id="Hello", secret_key="World")
```

**2. Load credentials from `.json` file with `load_key_from_json`.**

JSON file should have the following format:
```json
{"pub_id": "PUBLIC_ID", "secret_key": "SECRET_KEY"}
```
We recommend loading from `.json`. With this method you only have to save your credentials in one (safe) place and avoid leaving reference to a secret key in Python code.

```py
from numerblox.misc import load_key_from_json
key = load_key_from_json("my_credentials.json")
```

## Numerai Classic

Submissions can be done in 2 lines of code. To initialize the submitter object, pass a directory path for saving submissions and a `Key` object.

`NumeraiClassicSubmitter.full_submission` will perform:
 1. Checks to prevent surprise behavior (including value range and column validity)
 2. Saving to CSV
 3. Uploading with `numerapi`.

The `dataf` argument can be either a `pd.DataFrame` or `NumerFrame`.

```py
from numerblox.submission import NumeraiClassicSubmitter
submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key)
# Your prediction file with 'id' as index and defined 'cols' below.
dataf = pd.DataFrame(columns=["prediction"])
# Only works with valid key credentials and model_name
submitter.full_submission(dataf=dataf,
                          cols="prediction",
                          file_name="submission.csv",
                          model_name="my_model")
```

## Numerai Signals

`NumeraiSignalsSubmitter` is very similar to `NumeraiClassicSubmitter`, but has a few additional checks specific to Signals. Mainly, it checks if the data contains a valid ticker column (`"cusip"`, `"sedol"`, `"ticker"`, `"numerai_ticker"` or `"bloomberg_ticker"`) and a `'signal'` column.

`NumeraiSignalsSubmitter.full_submission` handles checks, saving of CSV and uploading with `numerapi`.

```py
from numerblox.submission import NumeraiSignalsSubmitter
submitter = NumeraiSignalsSubmitter(directory_path="sub_current_round", key=key)
# Your prediction file with 'id' as index, a valid ticker column and signal column below.
dataf = pd.DataFrame(columns=['bloomberg_ticker', 'signal'])
# Only works with valid key credentials and model_name
submitter.full_submission(dataf=dataf,
                          cols=["bloomberg_ticker", "signal"],
                          file_name="submission.csv",
                          model_name="my_model")
```

## NumerBay

NumerBlox also offers functionality to submit predictions from [NumerBay](https://numerbay.ai). This is a marketplace where Numerai predictions are bought and sold. Uploading from Numerbay is similar, but also requires authentication with your NumerBay account.

Also make sure the `numerbay` library is installed.

```bash
pip install numerbay
```

```py
from numerblox.submission import NumeraiClassicSubmitter, NumerBaySubmitter
# Your prediction DataFrame
dataf = pd.DataFrame(columns=["prediction"])

# Full submission to both Numerai and NumerBay
numerbay_submitter = NumerBaySubmitter(
    tournament_submitter = NumeraiClassicSubmitter(directory_path="sub_current_round", key=key),
    numerbay_username="yourusername",
    numerbay_password="yourpassword"
)
numerbay_submitter.full_submission(
    dataf=dataf,
    model_name="my_model",
    numerbay_product_full_name="numerai-predictions-yourproductname",
    file_name="submission.csv"
)
```

## Note

When you are done with submissions and don't need the submission file you can remove the submission directory with 1 line. Convenient if you have automated jobs and want to avoid clutter due to saving submission files for every round.

```py
# Clean up environment
submitter.remove_base_directory()
```
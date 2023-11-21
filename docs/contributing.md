# How To Contribute

First, thank you for your consideration to contribute to `numerblox`! This document provides some general guidelines to streamline the contribution process.

## Installation

If you haven't installed `numerblox` yet, clone the project into your favorite development environment and install the repo in editable mode and with all dev dependencies. 

```bash
git clone https://github.com/crowdcent/numerblox.git
pip install poetry
cd numerblox
poetry install
```

## Developing considerations

### 1. Building a new component

If you want to build a new component. Please consider the following steps:

1. Place the new component in the appropriate section. Is it a Downloader (`download.py`), a Preprocessor (`preprocessing.py`) or a Submitting tool (`submission.py`)? Also check the documentation on that section to check for templates, conventions and how these blocks are constructed in general.
2. Add tests for this new component in the appropriate test file. If you are introducing a new Downloader, add tests in `tests/test_downloader.py`. If you are introducing a new Preprocessor, add tests in `tests/test_preprocessing.py`. etc.
3. When making a preprocessor or postprocessor, make sure the component follows [scikit-learn conventions](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator). The core things to implement are inheriting from `BaseEstimator` and implementing `fit`, `transform` and `get_feature_names_out` methods. 
4. If your component introduces new dependencies, make sure to add them to poetry with `poetry add <library>`.


### 2. Fixing bugs
Even though most of the components in this library are tested, users will still likely run into issues. If you discover bugs, other issues or ideas for enhancements, do not hesitate to make a Github issue. Describe in the issue what code was run on what machine and background on the issue. Add stacktraces and screenshots if this is relevant for solving the issue. Also, please add appropriate labels for the Github issue.

- Ensure the bug was not already reported by searching on GitHub under Issues.
- If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
- Be sure to add the complete error messages.
- Be sure to add tests that fail without your patch, and pass with it.

### 3. Creating an example notebook

We welcome example notebooks that demonstrate the use of `numerblox`. If you want to create an example notebook, please make a notebook in the `examples/` folder. Make sure to add appropriate descriptions and explain the process of using the various components. Before committing please run the notebook from top to bottom. If it runs without errors, you can commit the notebook.
Lastly, if the notebook uses additional libraries, please note this at the top of the notebook and create a code block with `!pip install <library>`.

Example pip install cell:

```bash
!pip install scikit-lego plotly
```

#### Did you write a patch that fixes a bug?
- Open a new GitHub pull request with the patch.
- Ensure that your PR includes a test that fails without your patch, and pass with it.
- Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## PR submission guidelines
- Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
- Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
- If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

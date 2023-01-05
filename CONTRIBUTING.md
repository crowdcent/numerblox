# How to contribute

We are using a project setup called [nbdev](https://nbdev.fast.ai/) to easily develop code, documentation and tests within Jupyter notebooks. If you are using the library and have no interest in contributing, don't worry about this. Just `pip install numerblox` and you are good to go!

Else, thanks a lot for wanting to help us out with this project! If you are thinking of contributing and are not familiar with `nbdev`, it may take some time to learn nbdev development. 
We are happy to help out and point you to documentation or videos to learn more.

If you are interested in the full scope of what nbdev has to offer, check out this tutorial by Jeremy Howard:
[https://nbdev.fast.ai/tutorials/tutorial.html](https://nbdev.fast.ai/tutorials/tutorial.html).

Why are we using nbdev? To learn more about the rationale behind nbdev:
[https://youtu.be/9Q6sLbz37gk](https://youtu.be/9Q6sLbz37gk)

## How to get started

Before anything else, please install the git hooks that run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata (and avoid merge conflicts). After cloning the repository, run the following command inside it:
```
nbdev_install_git_hooks
```

## Did you find a bug?

Even though most of the components in this library are tested, users will still likely run into issues. If you discover bugs, other issues or ideas for enhancements, do not hesitate to make a Github issue. Describe in the issue what code was run on what machine and background on the issue. Add stacktraces and screenshots if this is relevant for solving the issue. Also, please add appropriate labels for the Github issue.

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes a test that fails without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

## Contributing Code

There are a few small things you should do before contributing code to this project. After you clone the repository, please run `nbdev_install_git_hooks` in your terminal. This sets up git hooks, which cleans up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran). This avoids unnecessary merge conflicts.

When adding a new feature, only change code in the `nbs/` directory. Then, before pushing code, be sure to run `nbdev_prepare`. `nbdev` automatically handles the parsing of notebook code into source code and automatically generates documentation from these notebooks.

## Do you want to contribute example notebooks?

Same guidelines as "Contributing Code" section. New notebooks should be created in the `nbs/edu_nbs` directory.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the nbs folder.
* Therefore, to change the documentation, only change files in the `nbs/` folder. Then, before pushing new documentation, be sure to run `nbdev_prepare` so the notebooks and source code are synced. `nbdev` automatically generates documentation from these notebooks and uploads to Github pages.

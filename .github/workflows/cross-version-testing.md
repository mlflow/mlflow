# Cross version testing

## What is cross version testing?

Cross version testing is a testing strategy to ensure ML integrations in MLflow such as
`mlflow.sklearn` work properly with their associated packages across various versions.

## Key files

| File (relative path from the root)              | Role                                                           |
| :---------------------------------------------- | :------------------------------------------------------------- |
| [`mlflow/ml-package-versions.yml`][]            | Define which versions to test for each ML package.             |
| [`dev/set_matrix.py`][]                         | Generate a test matrix from `ml-package-versions.yml`.         |
| [`dev/update_ml_package_versions.py`][]         | Update `ml-package-versions.yml` when releasing a new version. |
| [`.github/workflows/cross-version-tests.yml`][] | Define a Github Actions workflow for cross version testing.    |

[`mlflow/ml-package-versions.yml`]: ../../mlflow/ml-package-versions.yml
[`dev/set_matrix.py`]: ../../dev/set_matrix.py
[`dev/update_ml_package_versions.py`]: ../../dev/update_ml_package_versions.py
[`.github/workflows/cross-version-tests.yml`]: ./cross-version-tests.yml

## Configuration keys in `ml-package-versions.yml`

```yml
# Note this is just an example and not the actual sklearn configuration.

# The top-level key specifies the integration name.
sklearn:
  package_info:
    # [Required] `pip_release` specifies the package this integration depends on.
    pip_release: "scikit-learn"

    # [Optional] `install_dev` specifies a set of commands to install the dev version of the package.
    # For example, the command below builds a wheel from the latest main branch of
    # the scikit-learn repository and installs it.
    #
    # The aim of testing the dev version is to spot issues as early as possible before they get
    # piled up, and fix them incrementally rather than fixing them at once when the package
    # releases a new version.
    install_dev: |
      pip install git+https://github.com/scikit-learn/scikit-learn.git

  # [At least one of `models` and `autologging` must be specified]
  # `models` specifies the configuration for model serialization and serving tests.
  # `autologging` specifies the configuration for autologging tests.
  models or autologging:
    # [Optional] `requirements` specifies additional pip requirements required for running tests.
    # For example, '">= 0.24.0": ["xgboost"]' is interpreted as 'if the version of scikit-learn
    # to install is newer than or equal to 0.24.0, install xgboost'.
    requirements:
      ">= 0.24.0": ["xgboost"]

    # [Required] `minimum` specifies the minimum supported version for the latest release of MLflow.
    minimum: "0.20.3"

    # [Required] `maximum` specifies the maximum supported version for the latest release of MLflow.
    maximum: "1.0"

    # [Optional] `unsupported` specifies a list of versions that should NOT be supported due to
    # unacceptable issues or bugs.
    unsupported: ["0.21.3"]

    # [Required] `run` specifies a set of commands to run tests.
    run: |
      pytest tests/sklearn/test_sklearn_model_export.py
```

## How do we determine which versions to test?

We determine which versions to test based on the following rules:

1. Only test [final][] (e.g. `1.0.0`) and [post][] (`1.0.0.post0`) releases.
2. Only test the latest micro version in each minor version.
   For example, if `1.0.0`, `1.0.1`, and `1.0.2` are available, we only test `1.0.2`.
3. The `maximum` version defines the maximum **major** version to test.
   For example, if the value of `maximum` is `1.0.0`, we test `1.1.0` (if available) but not `2.0.0`.
4. Always test the `minimum` version.

[final]: https://www.python.org/dev/peps/pep-0440/#final-releases
[post]: https://www.python.org/dev/peps/pep-0440/#post-releases

The table below describes which `scikit-learn` versions to test for the example configuration in
the previous section:

| Version       | Tested | Comment                                            |
| :------------ | :----- | -------------------------------------------------- |
| 0.20.3        | ✅     | The value of `minimum`                             |
| 0.20.4        | ✅     | The latest micro version of `0.20`                 |
| 0.21rc2       |        |                                                    |
| 0.21.0        |        |                                                    |
| 0.21.1        |        |                                                    |
| 0.21.2        | ✅     | The latest micro version of `0.21` without`0.21.3` |
| 0.21.3        |        | Excluded by `unsupported`                          |
| 0.22rc2.post1 |        |                                                    |
| 0.22rc3       |        |                                                    |
| 0.22          |        |                                                    |
| 0.22.1        |        |                                                    |
| 0.22.2        |        |                                                    |
| 0.22.2.post1  | ✅     | The latest micro version of `0.22`                 |
| 0.23.0rc1     |        |                                                    |
| 0.23.0        |        |                                                    |
| 0.23.1        |        |                                                    |
| 0.23.2        | ✅     | The latest micro version of `0.23`                 |
| 0.24.dev0     |        |                                                    |
| 0.24.0rc1     |        |                                                    |
| 0.24.0        |        |                                                    |
| 0.24.1        |        |                                                    |
| 0.24.2        | ✅     | The latest micro version of `0.24`                 |
| 1.0rc1        |        |                                                    |
| 1.0rc2        |        |                                                    |
| 1.0           |        | The value of `maximum`                             |
| 1.0.1         | ✅     | The latest micro version of `1.0`                  |
| 1.1.dev       | ✅     | The version installed by `install_dev`             |

## Why do we run tests against development versions?

In cross-version testing, we run daily tests against both publicly available and pre-release
development versions for all dependent libraries that are used by MLflow.
This section explains why.

### Without dev version test

First, let's take a look at what would happen **without** dev version test.

```
  |
  ├─ XGBoost merges a change on the master branch that breaks MLflow's XGBoost integration.
  |
  ├─ MLflow 1.20.0 release date
  |
  ├─ XGBoost 1.5.0 release date
  ├─ ❌ We notice the change here and might need to make a patch release if it's critical.
  |
  v
time
```

- We didn't notice the change until after XGBoost 1.5.0 was released.
- MLflow 1.20.0 doesn't work with XGBoost 1.5.0.

### With dev version test

Then, let's take a look at what would happen **with** dev version test.

```
  |
  ├─ XGBoost merges a change on the master branch that breaks MLflow's XGBoost integration.
  ├─ ✅ Tests for the XGBoost integration fail -> We can notice the change and apply a fix for it.
  |
  ├─ MLflow 1.20.0 release date
  |
  ├─ XGBoost 1.5.0 release date
  |
  v
time
```

- We can notice the change **before XGBoost 1.5.0 is released** and apply a fix for it **before releasing MLflow 1.20.0**.
- MLflow 1.20.0 works with XGBoost 1.5.0.

## When do we run cross version tests?

1. Daily at 7:00 UTC using a cron scheduler.
   [README on the repository root](../../README.md) has a badge ([![badge-img][]][badge-target]) that indicates the status of the most recent cron run.
2. When a PR that affects the ML integrations is created. Note we only run tests relevant to
   the affected ML integrations. For example, a PR that affects files in `mlflow/sklearn` triggers
   cross version tests for `sklearn`.

[badge-img]: https://github.com/mlflow/mlflow/workflows/Cross%20version%20tests/badge.svg?event=schedule
[badge-target]: https://github.com/mlflow/mlflow/actions?query=workflow%3ACross%2Bversion%2Btests+event%3Aschedule

## How to run cross version test for dev versions on a pull request

By default, cross version tests for dev versions are disabled on a pull request.
To enable them, the following steps are required.

1. Click `Labels` in the right sidebar.
2. Click the `enable-dev-tests` label and make sure it's applied on the pull request.
3. Push a new commit or re-run the `cross-version-tests` workflow.

See also:

- [GitHub Docs - Applying a label](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels#applying-a-label)
- [GitHub Docs - Re-running workflows and jobs](https://docs.github.com/en/actions/managing-workflow-runs/re-running-workflows-and-jobs)

## How to run cross version tests manually

The `cross-version-tests.yml` workflow can be run manually without creating a pull request.

1. Open https://github.com/mlflow/mlflow/actions/workflows/cross-version-tests.yml.
2. Click `Run workflow`.
3. Fill in the input parameters.
4. Click `Run workflow` at the bottom of the parameter input form.

See also:

- [GitHub Docs - Manually running a workflow](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow)

# Cross version testing

## What is cross version testing?

Cross version testing is a testing strategy we adopt to ensure ML integrations in MLflow
(e.g. `mlflow.sklearn`) work properly with their associated packages across different versions.

## Key files

| File                                        | Role                                                           |
| :------------------------------------------ | :------------------------------------------------------------- |
| `mlflow/ml-package-versions.yml`            | Define which versions to test for each ML package.             |
| `dev/set_matrix.py`                         | Generate a test matrix from `ml-package-versions.yml`.         |
| `dev/update_ml_package_versions.py`         | Update `ml-package-versions.yml` when releasing a new version. |
| `.github/workflows/cross-version-tests.yml` | Define a Github Actions workflow for cross version testing.    |

## Configuration keys in `ml-package-versions.yml`

```yml
# The top-level key specifies the integration name.
sklearn:
  package_info:
    # [Required] `pip_release` specifies the package this integration depends on.
    pip_release: "scikit-learn"

    # [Optional] `install_dev` specifies a set of commands to install the dev version of the package.
    # For example, the command below builds a wheel from the latest main branch of
    # the scikit-learn repository and installs it.
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
    # Our CI will still test all the way up through the current maximum version that's been released
    # on PyPI. For example, if scikit-learn 1.0.1 is available on PyPI, our CI will pick it up.
    maximum: "1.0"

    # [Optional] `unsupported` specifies a list of versions that should NOT be supported due to
    # unacceptable issues or bugs.
    unsupported: ["0.21.3"]

    # [Required] `run` specifies a set of commands to run tests.
    run: |
      pytest tests/sklearn/test_sklearn_model_export.py --large
```

## How do we determine which versions to test?

We determine which versions to test by filtering candidates between `minimum` and `maximum` based
on the following rules:

1. Only test the latest micro version of each minor version.
2. Skip [pre-releases](https://www.python.org/dev/peps/pep-0440/#pre-releases) (e.g. `1.0rc1`).
3. Always test the `minimum` version.

The table below describes which `scikit-learn` versions to test for the example configuration in
the previous section:

| Version       | Tested | Comment                                |
| :------------ | :----- | -------------------------------------- |
| 0.20.3        | ✅     | The value of `minimum`                 |
| 0.20.4        | ✅     | The latest micro version of `0.20`     |
| 0.21rc2       |        |                                        |
| 0.21.0        |        |                                        |
| 0.21.1        |        |                                        |
| 0.21.2        |        |                                        |
| 0.21.3        |        | Excluded by `unsupported`              |
| 0.22rc2.post1 |        |                                        |
| 0.22rc3       |        |                                        |
| 0.22          |        |                                        |
| 0.22.1        |        |                                        |
| 0.22.2        |        |                                        |
| 0.22.2.post1  | ✅     | The latest micro version of `0.22`     |
| 0.23.0rc1     |        |                                        |
| 0.23.0        |        |                                        |
| 0.23.1        |        |                                        |
| 0.23.2        | ✅     | The latest micro version of `0.23`     |
| 0.24.dev0     |        |                                        |
| 0.24.0rc1     |        |                                        |
| 0.24.0        |        |                                        |
| 0.24.1        |        |                                        |
| 0.24.2        | ✅     | The latest micro version of `0.24`     |
| 1.0rc1        |        |                                        |
| 1.0rc2        |        |                                        |
| 1.0           |        | The value of `maximum`                 |
| 1.0.1         | ✅     | The latest micro version of `1.0`      |
| 1.1.dev       | ✅     | The version installed by `install_dev` |

## When do we run cross version tests?

1. Daily at 7:00 UTC using a cron scheduler.
2. When a PR that affects the ML integrations is filed. Note we only run tests relevant to
   the affected ML integrations. For example, a PR that affects files in `mlflow/sklearn` triggers
   cross version tests for `sklearn`.

## How to run dev tests on a pull request

Steps:

1. Click `Labels` in the right sidebar.
2. Select the `enable-dev-tests` label and make sure it's applied.
3. Push a new commit or re-run the `cross-version-tests` workflow.

See also:

- https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels#applying-a-label
- https://docs.github.com/en/actions/managing-workflow-runs/re-running-workflows-and-jobs

## How to run cross version tests manually

Steps:

1. Open https://github.com/mlflow/mlflow/actions/workflows/cross-version-tests.yml.
2. Select `Run workflow`.
3. Fill in the required input parameters.
4. Click `Run workflow` at the bottom.

See also:

- https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow

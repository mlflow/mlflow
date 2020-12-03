# Cross version tests

## Files & Roles

### `ml-package-versions.yml`

Define package versions to test and dependencies required to run the tests for each flavor.

### `dev/set_matrix.py`

Read `ml-package-versions.yml` and set a test matrix.

### `cross-versions-tests.yml`

Run the following two jobs.

- `set-matrix`: Run `set_matrix.py` to set a test matrix.
- `test`: Sweep the test matrix set by `set-matrix`.

## When do we run the cross version tests?

1. On a pull request: Run tests affected by the PR
2. On a daily scheduled job (at 7:00 UTC): Run all tests

## How to run `dev/set_matrix.py`

```sh
# ===== Include all items =====

python dev/set_matrix.py

# ===== Include only `ml-package-versions.yml` updates =====

REF_VERSIONS_YAML="https://raw.githubusercontent.com/mlflow/mlflow/master/ml-package-versions.yml"
python dev/set_matrix.py --ref-versions-yaml $REF_VERSIONS_YAML

# ===== Include only flavor file updates =====

CHANGED_FILES="
mlflow/keras.py
tests/xgboost/test_xgboost_autolog.py
"
python dev/set_matrix.py --changed-files $CHANGED_FILES

# ===== Include both `ml-package-versions.yml` & flavor file updates =====

python dev/set_matrix.py --ref-versions-yaml $REF_VERSIONS_YAML --changed-files $CHANGED_FILES
```

## How to run doctests in `dev/set_matrix.py`

```
pytest dev/set_matrix.py --doctest-modules --verbose
```

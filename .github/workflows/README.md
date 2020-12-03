# Cross version tests

## Files & Roles

| File (relative path from the root)          | Role                                                                                                                |
| :------------------------------------------ | :------------------------------------------------------------------------------------------------------------------ |
| `ml-package-versions.yml`                   | Define package versions to test and dependencies required to run tests for each flavor                              |
| `dev/set_matrix.py`                         | Read `ml-package-versions.yml` and set a test matrix                                                                |
| `.github/workflows/cross-version-tests.yml` | Run tests across multiple combinations of packages and versions based on the test matrix set by `dev/set_matrix.py` |

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

#!/usr/bin/env bash

ALLOWED_PATTERNS='Mlflow\(|"Mlflow"|import Mlflow$'
# add globs to this list to ignore them in grep
EXCLUDED_FILES=(
    # ignore typos in i18n files, since they're not controlled by us
    "mlflow/server/js/src/lang/*.json"
    "mlflow/server/js/src/common/utils/StringUtils.ts"
    "dev/clint/tests/rules/test_mlflow_class_name.py"
)

EXCLUDE_ARGS=""
for pattern in "${EXCLUDED_FILES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pattern"
done

if grep -InE ' \bM(lf|LF|lF)low\b' $EXCLUDE_ARGS "$@" | grep -vE "$ALLOWED_PATTERNS"; then
    echo -e "\nFound typo for MLflow spelling in above file(s). Please use 'MLflow' instead of 'Mlflow'."
    exit 1
else
    exit 0
fi

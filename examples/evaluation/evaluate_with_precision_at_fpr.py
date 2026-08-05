"""
Example: Model selection using precision at a fixed false-positive rate (FPR).

In fraud detection and other high-stakes classification tasks, raw accuracy
is a misleading criterion. A model that flags everything as legitimate will
score well on accuracy but catch no fraud. The metric that matters is
precision at a controlled FPR — how many flagged transactions are genuine
fraud, given a cap on how often we interrupt legitimate transactions.

This example shows how to:
- Define a custom metric for precision at a fixed FPR using make_metric
- Use MetricThreshold to gate model registration on that metric
- Compare a candidate XGBoost model against a baseline using
  mlflow.validate_evaluation_results
"""

import numpy as np
import xgboost
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import MetricThreshold, infer_signature, make_metric


# ── Dataset ──────────────────────────────────────────────────────────────────
# Synthetic imbalanced dataset mimicking a fraud detection scenario:
# 5% positive class (fraudulent transactions)
X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=10,
    weights=[0.95, 0.05],
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ── Models ────────────────────────────────────────────────────────────────────
candidate_model = xgboost.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    scale_pos_weight=19,  # compensate for class imbalance
    eval_metric="logloss",
    random_state=42,
).fit(X_train, y_train)
candidate_signature = infer_signature(X_train, candidate_model.predict(X_train))

baseline_model = DummyClassifier(strategy="stratified").fit(X_train, y_train)
baseline_signature = infer_signature(X_train, baseline_model.predict(X_train))

# ── Eval data ─────────────────────────────────────────────────────────────────
import pandas as pd

eval_data = pd.DataFrame(X_test)
eval_data["label"] = y_test


# ── Custom metric: precision at fixed FPR ────────────────────────────────────
TARGET_FPR = 0.003  # tolerate at most 0.3% false positive rate


def precision_at_target_fpr(eval_df, builtin_metrics):
    """
    Precision achieved when the decision threshold is chosen so that the
    false-positive rate does not exceed TARGET_FPR.

    Higher precision at a fixed FPR means the model surfaces real fraud
    without over-flagging legitimate transactions.
    """
    from sklearn.metrics import roc_curve

    y_true = eval_df["target"]
    y_score = eval_df["prediction"]

    # roc_curve returns fpr, tpr, thresholds in decreasing threshold order
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find the highest threshold where FPR is still within budget
    valid = thresholds[fpr <= TARGET_FPR]
    if len(valid) == 0:
        return 0.0

    chosen_threshold = valid[0]
    y_pred = (y_score >= chosen_threshold).astype(int)

    return float(precision_score(y_true, y_pred, zero_division=0))


precision_at_fpr_metric = make_metric(
    eval_fn=precision_at_target_fpr,
    greater_is_better=True,
    name="precision_at_target_fpr",
)

# ── Thresholds ────────────────────────────────────────────────────────────────
thresholds = {
    # Candidate must achieve at least 0.60 precision at the 0.3% FPR budget
    "precision_at_target_fpr": MetricThreshold(
        threshold=0.60,
        greater_is_better=True,
    ),
    # Candidate must also beat the baseline on standard precision
    "precision_score": MetricThreshold(
        min_absolute_change=0.05,
        min_relative_change=0.05,
        greater_is_better=True,
    ),
}

# ── Run ───────────────────────────────────────────────────────────────────────
with mlflow.start_run() as run:
    baseline_model_uri = mlflow.sklearn.log_model(
        baseline_model,
        name="baseline_model",
        signature=baseline_signature,
        serialization_format="cloudpickle",
    ).model_uri

    baseline_result = mlflow.evaluate(
        baseline_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        extra_metrics=[precision_at_fpr_metric],
        env_manager="local",
    )

    candidate_model_uri = mlflow.sklearn.log_model(
        candidate_model,
        name="candidate_model",
        signature=candidate_signature,
        serialization_format="cloudpickle",
    ).model_uri

    candidate_result = mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        extra_metrics=[precision_at_fpr_metric],
        env_manager="local",
    )

# ── Validate ──────────────────────────────────────────────────────────────────
mlflow.validate_evaluation_results(
    candidate_result=candidate_result,
    baseline_result=baseline_result,
    validation_thresholds=thresholds,
)
# ModelValidationFailedException is raised if thresholds are not met.
# Wrap mlflow.validate_evaluation_results in a try/except block to handle
# this gracefully in CI/CD pipelines.

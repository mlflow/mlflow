import shap
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import MetricThreshold, infer_signature, make_metric

# load UCI Adult Data Set; segment it into training and test sets
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train a candidate XGBoost model
candidate_model = xgboost.XGBClassifier().fit(X_train, y_train)
candidate_signature = infer_signature(X_train, candidate_model.predict(X_train))

# train a baseline dummy model
baseline_model = DummyClassifier(strategy="uniform").fit(X_train, y_train)
baseline_signature = infer_signature(X_train, baseline_model.predict(X_train))

# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["label"] = y_test


# Define a custom metric to evaluate against
def double_positive(_eval_df, builtin_metrics):
    return builtin_metrics["true_positives"] * 2


# Define criteria for model to be validated against
thresholds = {
    # Specify metric value threshold
    "precision_score": MetricThreshold(
        threshold=0.7, greater_is_better=True
    ),  # precision should be >=0.7
    # Specify model comparison thresholds
    "recall_score": MetricThreshold(
        min_absolute_change=0.1,  # recall should be at least 0.1 greater than baseline model recall
        min_relative_change=0.1,  # recall should be at least 10 percent greater than baseline model recall
        greater_is_better=True,
    ),
    # Specify both metric value and model comparison thresholds
    "accuracy_score": MetricThreshold(
        threshold=0.8,  # accuracy should be >=0.8
        min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=True,
    ),
    # Specify threshold for custom metric
    "double_positive": MetricThreshold(
        threshold=1e5,
        greater_is_better=False,  # double_positive should be <=1e5
    ),
}

double_positive_metric = make_metric(
    eval_fn=double_positive,
    greater_is_better=False,
)

with mlflow.start_run() as run:
    # Note: in most model validation use-cases the baseline model should instead b
    # a previously trained model (such as the current production model)
    baseline_model_uri = mlflow.sklearn.log_model(
        baseline_model, name="baseline_model", signature=baseline_signature
    ).model_uri

    # Evaluate the baseline model
    baseline_result = mlflow.evaluate(
        baseline_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        extra_metrics=[double_positive_metric],
        # set to env_manager to "virtualenv" or "conda" to score the candidate and baseline models
        # in isolated Python environments where their dependencies are restored.
        env_manager="local",
    )

    # Evaluate the candidate model
    candidate_model_uri = mlflow.sklearn.log_model(
        candidate_model, name="candidate_model", signature=candidate_signature
    ).model_uri

    candidate_result = mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        extra_metrics=[double_positive_metric],
        env_manager="local",
    )


# Validate the candidate result against the baseline
mlflow.validate_evaluation_results(
    candidate_result=candidate_result,
    baseline_result=baseline_result,
    validation_thresholds=thresholds,
)
# If you would like to catch model validation failures, you can add try except clauses around
# the mlflow.evaluate() call and catch the ModelValidationFailedException, imported at the top
# of this file.

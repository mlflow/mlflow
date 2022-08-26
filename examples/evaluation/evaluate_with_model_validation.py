import xgboost
import shap
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import mlflow
from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

# load UCI Adult Data Set; segment it into training and test sets
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train a candidate XGBoost model
candidate_model = xgboost.XGBClassifier().fit(X_train, y_train)

# train a baseline dummy model
baseline_model = DummyClassifier(strategy="uniform").fit(X_train, y_train)

# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["label"] = y_test

# Define a custom metric to evaluate against
def double_positive(_, builtin_metrics):
    return {
        "double_positive": builtin_metrics["true_positives"] * 2,
    }


# Define criteria for model to be validated against
thresholds = {
    # Specify metric value threshold
    "precision": MetricThreshold(threshold=0.7, higher_is_better=True),  # precision should be >=0.7
    # Specify model comparison thresholds
    "recall": MetricThreshold(
        min_absolute_change=0.1,  # recall should be at least 0.1 greater than baseline model recall
        min_relative_change=0.1,  # recall should be at least 10 percent greater than baseline model recall
        higher_is_better=True,
    ),
    # Specify both metric value and model comparison thresholds
    "accuracy": MetricThreshold(
        threshold=0.8,  # accuracy should be >=0.8
        min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        higher_is_better=True,
    ),
    # Specify threshold for custom metric
    "double_positive": MetricThreshold(
        threshold=1e5, higher_is_better=False  # double_positive should be <=1e5
    ),
}

with mlflow.start_run() as run:
    candidate_model_uri = mlflow.sklearn.log_model(candidate_model, "candidate_model").model_uri
    # Note: in most model validation use-cases the baseline model should instead be a previously
    # trained model (such as the current production model), specified directly in the
    # mlflow.evaluate() call via model URI.
    baseline_model_uri = mlflow.sklearn.log_model(baseline_model, "baseline_model").model_uri

    try:
        mlflow.evaluate(
            candidate_model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            dataset_name="adult",
            evaluators=["default"],
            validation_thresholds=thresholds,
            custom_metrics=[double_positive],
            baseline_model=baseline_model_uri,
        )
    except ModelValidationFailedException as e:
        print("Model failed to pass validation criteria! See detailed error message:")
        print(e)
    else:
        print("Model passed validation criteria!")

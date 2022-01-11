import xgboost
import shap
from mlflow.models.evaluation import evaluate, EvaluationDataset
import mlflow
from sklearn.model_selection import train_test_split

# train XGBoost model
X, y = shap.datasets.adult()

num_examples = len(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = xgboost.XGBClassifier().fit(X_train, y_train)

eval_data = X_test
eval_data['label'] = y_test

eval_dataset = EvaluationDataset(data=eval_data, labels='label', name='adult')

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, 'model')
    model_uri = mlflow.get_artifact_uri('model')
    result = evaluate(
        model=model_uri,
        model_type='classifier',
        dataset=eval_dataset,
        evaluators=['default'],
    )

print(f'metrics:\n{result.metrics}')
print(f'artifacts:\n{result.artifacts}')

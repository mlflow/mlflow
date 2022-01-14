import mlflow
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

boston_data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    boston_data.data, boston_data.target, test_size=0.33, random_state=42
)

with mlflow.start_run() as run:
    model = LinearRegression().fit(X_train, y_train)
    model_uri = mlflow.get_artifact_uri('model')

    result = mlflow.evaluate(
        model_uri,
        X_test,
        targets=y_test,
        model_type='regressor',
        dataset_name='boston',
        evaluators='default',
        feature_names=boston_data.feature_names,
        evaluator_config={
            'explainability_nsamples': 1000
        }
    )

print(f'metrics:\n{result.metrics}')
print(f'artifacts:\n{result.artifacts}')

import numpy as np
import json

from mlflow.models.evaluation import evaluate, EvaluationDataset
import mlflow
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from tests.models.test_evaluation import get_run_data, \
    regressor_model_uri, diabetes_dataset, \
    classifier_model_uri, iris_dataset, \
    binary_classifier_model_uri, breast_cancer_dataset


def test_regressor_evaluation(regressor_model_uri, diabetes_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            regressor_model_uri,
            model_type='regressor',
            dataset=diabetes_dataset,
            evaluators='default',
        )
        print(f'regressor evaluation run: {run.info.run_id}')

    params, metrics, tags, artifacts = \
        get_run_data(run.info.run_id)

    expected_metrics = {
        'example_count_on_data_diabetes_dataset': 148.0,
        'mean_absolute_error_on_data_diabetes_dataset': 42.927,
        'mean_squared_error_on_data_diabetes_dataset': 2747.513,
        'root_mean_squared_error_on_data_diabetes_dataset': 52.416,
        'sum_on_label_on_data_diabetes_dataset': 23099.0,
        'mean_on_label_on_data_diabetes_dataset': 156.074,
        'r2_score_on_data_diabetes_dataset': 0.565,
        'max_error_on_data_diabetes_dataset': 151.354,
        'mean_absolute_percentage_error_on_data_diabetes_dataset': 0.413
    }
    for metric_key in metrics:
        assert np.isclose(metrics[metric_key], expected_metrics[metric_key], rtol=1e-3)

    model = mlflow.pyfunc.load_model(regressor_model_uri)

    assert json.loads(tags['mlflow.datasets']) == \
        [{**diabetes_dataset._metadata, 'model': model.metadata.model_uuid}]

    assert set(artifacts) == {
        'shap_beeswarm_on_data_diabetes_dataset.png',
        'shap_feature_importance_on_data_diabetes_dataset.png',
        'shap_summary_on_data_diabetes_dataset.png',
    }


def test_multi_classifier_evaluation(classifier_model_uri, iris_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            classifier_model_uri,
            model_type='classifier',
            dataset=iris_dataset,
            evaluators='default',
        )
        print(f'multi-classifier evaluation run: {run.info.run_id}')


def test_bin_classifier_evaluation(binary_classifier_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            binary_classifier_model_uri,
            model_type='classifier',
            dataset=breast_cancer_dataset,
            evaluators='default',
        )

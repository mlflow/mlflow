import numpy as np
import json
import math
import sklearn.metrics

from mlflow.models.evaluation import evaluate, EvaluationDataset
from mlflow.models.evaluation.default_evaluator import _get_regressor_metrics
import mlflow
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from tests.models.test_evaluation import get_run_data, \
    regressor_model_uri, diabetes_dataset, \
    classifier_model_uri, iris_dataset, \
    binary_classifier_model_uri, breast_cancer_dataset, \
    spark_regressor_model_uri, diabetes_spark_dataset


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

    model = mlflow.pyfunc.load_model(regressor_model_uri)

    y = diabetes_dataset.labels
    y_pred = model.predict(diabetes_dataset.data)

    expected_metrics = _get_regressor_metrics(y, y_pred)
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + '_on_data_diabetes_dataset'],
            rtol=1e-3
        )
        assert np.isclose(
            expected_metrics[metric_key],
            result.metrics[metric_key],
            rtol=1e-3
        )

    assert json.loads(tags['mlflow.datasets']) == \
        [{**diabetes_dataset._metadata, 'model': model.metadata.model_uuid}]

    assert set(artifacts) == {
        'shap_beeswarm_plot_on_data_diabetes_dataset.png',
        'shap_feature_importance_plot_on_data_diabetes_dataset.png',
        'shap_summary_plot_on_data_diabetes_dataset.png',
    }
    assert result.artifacts.keys() == {
        'shap_beeswarm_plot',
        'shap_feature_importance_plot',
        'shap_summary_plot',
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

    params, metrics, tags, artifacts = \
        get_run_data(run.info.run_id)

    expected_metrics = {
        'accuracy': 0.32, 'example_count': 50, 'log_loss': 0.9712, 'f1_score_micro': 0.32,
        'f1_score_macro': 0.1616, 'class_0_true_negatives': 33,
        'class_0_false_positives': 0, 'class_0_false_negatives': 17, 'class_0_true_positives': 0,
        'class_0_recall': 0.0, 'class_0_precision': 0.0, 'class_0_f1_score': 0.0,
        'class_0_roc_auc': 1.0, 'class_0_precision_recall_auc': 1.0, 'class_1_true_negatives': 33,
        'class_1_false_positives': 0, 'class_1_false_negatives': 17, 'class_1_true_positives': 0,
        'class_1_recall': 0.0, 'class_1_precision': 0.0, 'class_1_f1_score': 0.0,
        'class_1_roc_auc': 0.9411, 'class_1_precision_recall_auc': 0.8989,
        'class_2_true_negatives': 0, 'class_2_false_positives': 34, 'class_2_false_negatives': 0,
        'class_2_true_positives': 16, 'class_2_recall': 1.0, 'class_2_precision': 0.32,
        'class_2_f1_score': 0.4848, 'class_2_roc_auc': 0.9963,
        'class_2_precision_recall_auc': 0.9921}

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + '_on_data_iris_dataset'],
            rtol=1e-3
        )
        assert np.isclose(
            expected_metrics[metric_key],
            result.metrics[metric_key],
            rtol=1e-3
        )

    model = mlflow.pyfunc.load_model(classifier_model_uri)

    assert json.loads(tags['mlflow.datasets']) == \
        [{**iris_dataset._metadata, 'model': model.metadata.model_uuid}]

    assert set(artifacts) == {
        'class_0_precision_recall_curve_data_on_data_iris_dataset.csv',
        'class_0_precision_recall_curve_plot_on_data_iris_dataset.png',
        'class_0_roc_curve_data_on_data_iris_dataset.csv',
        'class_0_roc_curve_plot_on_data_iris_dataset.png',
        'class_1_precision_recall_curve_data_on_data_iris_dataset.csv',
        'class_1_precision_recall_curve_plot_on_data_iris_dataset.png',
        'class_1_roc_curve_data_on_data_iris_dataset.csv',
        'class_1_roc_curve_plot_on_data_iris_dataset.png',
        'class_2_precision_recall_curve_data_on_data_iris_dataset.csv',
        'class_2_precision_recall_curve_plot_on_data_iris_dataset.png',
        'class_2_roc_curve_data_on_data_iris_dataset.csv',
        'class_2_roc_curve_plot_on_data_iris_dataset.png',
        'confusion_matrix_on_data_iris_dataset.png',
        'explainer_on_data_iris_dataset',
        'shap_beeswarm_plot_on_data_iris_dataset.png',
        'shap_feature_importance_plot_on_data_iris_dataset.png',
        'shap_summary_plot_on_data_iris_dataset.png',
    }
    assert result.artifacts.keys() == {
        'class_0_roc_curve_data', 'class_0_roc_curve_plot', 'class_0_precision_recall_curve_data',
        'class_0_precision_recall_curve_plot', 'class_1_roc_curve_data', 'class_1_roc_curve_plot',
        'class_1_precision_recall_curve_data', 'class_1_precision_recall_curve_plot',
        'class_2_roc_curve_data', 'class_2_roc_curve_plot', 'class_2_precision_recall_curve_data',
        'class_2_precision_recall_curve_plot', 'confusion_matrix', 'shap_beeswarm_plot',
        'shap_summary_plot', 'shap_feature_importance_plot'
    }


def test_bin_classifier_evaluation(binary_classifier_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            binary_classifier_model_uri,
            model_type='classifier',
            dataset=breast_cancer_dataset,
            evaluators='default',
        )
        print(f'bin-classifier evaluation run: {run.info.run_id}')

    params, metrics, tags, artifacts = \
        get_run_data(run.info.run_id)

    expected_metrics = {
        'accuracy': 0.957,
        'example_count': 190,
        'log_loss': 0.0918,
        'true_negatives': 71,
        'false_positives': 5,
        'false_negatives': 3,
        'true_positives': 111,
        'recall': 0.9736,
        'precision': 0.9568,
        'f1_score': 0.9652,
        'roc_auc': 0.995,
        'precision_recall_auc': 0.997
    }
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + '_on_data_breast_cancer_dataset'],
            rtol=1e-3
        )
        assert np.isclose(
            expected_metrics[metric_key],
            result.metrics[metric_key],
            rtol=1e-3
        )

    model = mlflow.pyfunc.load_model(binary_classifier_model_uri)

    assert json.loads(tags['mlflow.datasets']) == \
        [{**breast_cancer_dataset._metadata, 'model': model.metadata.model_uuid}]

    assert set(artifacts) == {
        'confusion_matrix_on_data_breast_cancer_dataset.png',
        'lift_curve_plot_on_data_breast_cancer_dataset.png',
        'precision_recall_curve_data_on_data_breast_cancer_dataset.csv',
        'precision_recall_curve_plot_on_data_breast_cancer_dataset.png',
        'roc_curve_data_on_data_breast_cancer_dataset.csv',
        'roc_curve_plot_on_data_breast_cancer_dataset.png',
        'shap_beeswarm_plot_on_data_breast_cancer_dataset.png',
        'shap_feature_importance_plot_on_data_breast_cancer_dataset.png',
        'shap_summary_plot_on_data_breast_cancer_dataset.png'
    }
    assert result.artifacts.keys() == {
        'roc_curve_data', 'roc_curve_plot', 'precision_recall_curve_data',
        'precision_recall_curve_plot', 'lift_curve_plot', 'confusion_matrix',
        'shap_beeswarm_plot', 'shap_summary_plot', 'shap_feature_importance_plot'
    }


def test_spark_model_evaluation(spark_regressor_model_uri, diabetes_spark_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            spark_regressor_model_uri,
            model_type='regressor',
            dataset=diabetes_spark_dataset,
            evaluators='default',
            evaluator_config={
                'log_model_explainability': True
            }
        )
        print(f'spark model evaluation run: {run.info.run_id}')

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)

    expected_metrics = {
        'example_count': 139.0,
        'mean_absolute_error': 45.672,
        'mean_squared_error': 3009.048,
        'root_mean_squared_error': 54.854,
        'sum_on_label': 21183.0,
        'mean_on_label': 152.395,
        'r2_score': 0.491,
        'max_error': 136.170,
        'mean_absolute_percentage_error': 0.41392110539896615
    }
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + '_on_data_diabetes_spark_dataset'],
            rtol=1e-3
        )
        assert np.isclose(
            expected_metrics[metric_key],
            result.metrics[metric_key],
            rtol=1e-3
        )

    model = mlflow.pyfunc.load_model(spark_regressor_model_uri)

    assert json.loads(tags['mlflow.datasets']) == \
        [{**diabetes_spark_dataset._metadata, 'model': model.metadata.model_uuid}]

    assert set(artifacts) == set()
    assert result.artifacts == {}

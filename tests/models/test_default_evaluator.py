import numpy as np
import json


from mlflow.models.evaluation import evaluate
from mlflow.models.evaluation.default_evaluator import (
    _get_classifier_global_metrics,
    _infer_model_type_by_labels,
    _extract_raw_model_and_predict_fn,
    _get_regressor_metrics,
    _get_binary_sum_up_label_pred_prob,
    _get_classifier_per_class_metrics,
    _gen_classifier_curve,
)
import mlflow
from sklearn.linear_model import LogisticRegression

# pylint: disable=unused-import
from tests.models.test_evaluation import (
    get_run_data,
    linear_regressor_model_uri,
    diabetes_dataset,
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
    svm_model_uri,
)
from mlflow.models.utils import plot_lines


def assert_dict_equal(d1, d2, rtol):
    for k in d1:
        assert k in d2
        assert np.isclose(d1[k], d2[k], rtol=rtol)


def test_regressor_evaluation(linear_regressor_model_uri, diabetes_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            model=linear_regressor_model_uri,
            model_type="regressor",
            dataset=diabetes_dataset,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels_data
    y_pred = model.predict(diabetes_dataset.features_data)

    expected_metrics = _get_regressor_metrics(y, y_pred)
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_diabetes_dataset.png",
        "shap_feature_importance_plot_on_data_diabetes_dataset.png",
        "shap_summary_plot_on_data_diabetes_dataset.png",
    }
    assert result.artifacts.keys() == {
        "shap_beeswarm_plot",
        "shap_feature_importance_plot",
        "shap_summary_plot",
    }


def test_multi_classifier_evaluation(multiclass_logistic_regressor_model_uri, iris_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            model=multiclass_logistic_regressor_model_uri,
            model_type="classifier",
            dataset=iris_dataset,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    _, _, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = iris_dataset.labels_data
    y_pred = predict_fn(iris_dataset.features_data)
    y_probs = predict_proba_fn(iris_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(False, y, y_pred, y_probs, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key], metrics[metric_key + "_on_data_iris_dataset"], rtol=1e-3
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**iris_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_iris_dataset.png",
        "per_class_metrics_on_data_iris_dataset.csv",
        "roc_curve_plot_on_data_iris_dataset.png",
        "precision_recall_curve_plot_on_data_iris_dataset.png",
        "shap_feature_importance_plot_on_data_iris_dataset.png",
        "explainer_on_data_iris_dataset",
        "confusion_matrix_on_data_iris_dataset.png",
        "shap_summary_plot_on_data_iris_dataset.png",
    }
    assert result.artifacts.keys() == {
        "per_class_metrics",
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_bin_classifier_evaluation(binary_logistic_regressor_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            model=binary_logistic_regressor_model_uri,
            model_type="classifier",
            dataset=breast_cancer_dataset,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, _, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, y_probs, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "lift_curve_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "precision_recall_curve_plot_on_data_breast_cancer_dataset.png",
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
        "roc_curve_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "lift_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_spark_regressor_model_evaluation(spark_linear_regressor_model_uri, diabetes_spark_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            model=spark_linear_regressor_model_uri,
            model_type="regressor",
            dataset=diabetes_spark_dataset,
            evaluators="default",
            evaluator_config={"log_model_explainability": True},
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X = diabetes_spark_dataset.features_data
    y = diabetes_spark_dataset.labels_data
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_spark_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_spark_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == set()
    assert result.artifacts == {}


def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            model=svm_model_uri,
            model_type="classifier",
            dataset=breast_cancer_dataset,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, _, predict_fn, _ = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, None, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_infer_model_type_by_labels():
    assert _infer_model_type_by_labels(["a", "b"]) == "classifier"
    assert _infer_model_type_by_labels([1, 2.5]) == "regressor"
    assert _infer_model_type_by_labels(list(range(2000))) == "regressor"
    assert _infer_model_type_by_labels([1, 2, 3]) == "classifier"


def test_extract_raw_model_and_predict_fn(binary_logistic_regressor_model_uri):
    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)
    (
        model_loader_module,
        raw_model,
        predict_fn,
        predict_proba_fn,
    ) = _extract_raw_model_and_predict_fn(model)
    assert model_loader_module == "mlflow.sklearn"
    assert isinstance(raw_model, LogisticRegression)
    assert predict_fn == raw_model.predict
    assert predict_proba_fn == raw_model.predict_proba


def test_get_regressor_metrics():
    y = [1.1, 2.1, -3.5]
    y_pred = [1.5, 2.0, -3.0]

    metrics = _get_regressor_metrics(y, y_pred)
    expected_metrics = {
        "example_count": 3,
        "mean_absolute_error": 0.3333333333333333,
        "mean_squared_error": 0.13999999999999999,
        "root_mean_squared_error": 0.3741657386773941,
        "sum_on_label": -0.2999999999999998,
        "mean_on_label": -0.09999999999999994,
        "r2_score": 0.976457399103139,
        "max_error": 0.5,
        "mean_absolute_percentage_error": 0.18470418470418468,
    }
    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


def test_get_binary_sum_up_label_pred_prob():
    y = [0, 1, 2]
    y_pred = [0, 2, 1]
    y_probs = [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5], [0.25, 0.4, 0.35]]

    results = []
    for idx, label in enumerate([0, 1, 2]):
        y_bin, y_pred_bin, y_prob_bin = _get_binary_sum_up_label_pred_prob(
            idx, label, y, y_pred, y_probs
        )
        results.append((list(y_bin), list(y_pred_bin), list(y_prob_bin)))

    print(results)
    assert results == [
        ([1, 0, 0], [1, 0, 0], [0.7, 0.2, 0.25]),
        ([0, 1, 0], [0, 0, 1], [0.1, 0.3, 0.4]),
        ([0, 0, 1], [0, 1, 0], [0.2, 0.5, 0.35]),
    ]


def test_get_classifier_per_class_metrics():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]

    expected_metrics = {
        "true_negatives": 3,
        "false_positives": 2,
        "false_negatives": 1,
        "true_positives": 4,
        "recall": 0.8,
        "precision": 0.6666666666666666,
        "f1_score": 0.7272727272727272,
    }
    metrics = _get_classifier_per_class_metrics(y, y_pred)
    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


def test_multiclass_get_classifier_global_metrics():
    y = [0, 1, 2, 1, 2]
    y_pred = [0, 2, 1, 1, 0]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    metrics = _get_classifier_global_metrics(
        is_binomial=False, y=y, y_pred=y_pred, y_probs=y_probs, labels=[0, 1, 2]
    )
    expected_metrics = {
        "accuracy": 0.4,
        "example_count": 5,
        "f1_score_micro": 0.4,
        "f1_score_macro": 0.38888888888888884,
        "log_loss": 1.1658691395263094,
    }
    assert_dict_equal(metrics, expected_metrics, 1e-3)


def test_binary_get_classifier_global_metrics():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]
    y_probs = [[1 - p, p] for p in y_prob]
    metrics = _get_classifier_global_metrics(
        is_binomial=True, y=y, y_pred=y_pred, y_probs=y_probs, labels=[0, 1]
    )
    expected_metrics = {"accuracy": 0.7, "example_count": 10, "log_loss": 0.6665822319387167}
    assert_dict_equal(metrics, expected_metrics, 1e-3)


def test_gen_binary_precision_recall_curve():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True, y=y, y_probs=y_prob, labels=[0, 1], curve_type="pr"
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.0]),
        rtol=1e-3,
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.55555556, 0.5, 0.57142857, 0.66666667, 0.6, 0.5, 0.66666667, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "recall"
    assert results.plot_fn_args["ylabel"] == "precision"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.7088888888888889, rtol=1e-3)


def test_gen_binary_roc_curve():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True, y=y, y_probs=y_prob, labels=[0, 1], curve_type="roc"
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([0.0, 0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0]),
        rtol=1e-3,
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.66, rtol=1e-3)


def test_gen_multiclass_precision_recall_curve():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False, y=y, y_probs=y_probs, labels=[0, 1, 2], curve_type="pr"
    )
    expected_x_data_list = [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.5, 0.0, 0.0]]
    expected_y_data_list = [
        [0.5, 0.0, 1.0],
        [0.66666667, 0.5, 1.0],
        [0.4, 0.25, 0.33333333, 0.5, 0.0, 1.0],
    ]
    line_labels = ["label=0,AP=0.500", "label=1,AP=0.722", "label=2,AP=0.414"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        assert np.allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        assert np.allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "recall"
    assert results.plot_fn_args["ylabel"] == "precision"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.25, 0.6666666666666666, 0.2875]
    assert np.allclose(results.auc, expected_auc, rtol=1e-3)


def test_gen_multiclass_roc_curve():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False, y=y, y_probs=y_probs, labels=[0, 1, 2], curve_type="roc"
    )
    print(results)

    expected_x_data_list = [
        [0.0, 0.25, 0.25, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0, 1.0],
    ]
    expected_y_data_list = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5, 1.0]]
    line_labels = ["label=0,AUC=0.750", "label=1,AUC=0.750", "label=2,AUC=0.333"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        assert np.allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        assert np.allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.75, 0.75, 0.3333]
    assert np.allclose(results.auc, expected_auc, rtol=1e-3)

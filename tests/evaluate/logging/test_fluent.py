import mlflow
import mlflow.evaluation


def test_fluent_evaluation_apis_are_available():
    assert mlflow.get_evaluation is mlflow.evaluation.get_evaluation
    assert mlflow.log_assessments is mlflow.evaluation.log_assessments
    assert mlflow.log_evaluation is mlflow.evaluation.log_evaluation
    assert mlflow.log_evaluations is mlflow.evaluation.log_evaluations
    assert mlflow.log_evaluations_df is mlflow.evaluation.log_evaluations_df
    assert mlflow.search_evaluations is mlflow.evaluation.search_evaluations
    assert mlflow.set_evaluation_tags is mlflow.evaluation.set_evaluation_tags

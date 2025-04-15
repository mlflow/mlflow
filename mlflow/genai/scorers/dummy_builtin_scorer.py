import copy

from mlflow.genai.scorers import BuiltInScorer


# TODO: ML-52304 example builtin scorer implementation
class DummyBuiltInScorer(BuiltInScorer):
    def update_evaluation_config(evaluation_config) -> dict:
        updated_config = copy.deepcopy(evaluation_config)
        updated_config["databricks-agents"]["metrics"].append("dummy_metric")
        return updated_config

from mlflow.genai.scorers import BuiltInScorer

# TODO: ML-52304 example builtin scorer implementation
class DummyBuiltInScorer(BuiltInScorer):
    def __call__(evaluation_config) -> dict:
        evaluation_config["databricks-agents"]["metrics"].append("dummy_metric")
    
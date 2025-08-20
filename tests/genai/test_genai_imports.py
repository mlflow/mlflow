import mlflow


def test_genai_imports():
    assert mlflow.genai.datasets.__name__ == 'mlflow.genai.datasets'
    assert mlflow.genai.evaluation.__name__ == 'mlflow.genai.evaluation'
    assert mlflow.genai.judges.__name__ == 'mlflow.genai.judges'
    assert mlflow.genai.labeling.__name__ == 'mlflow.genai.labeling'
    assert mlflow.genai.optimize.__name__ == 'mlflow.genai.optimize'
    assert mlflow.genai.prompts.__name__ == 'mlflow.genai.prompts'
    assert mlflow.genai.scorers.__name__ == 'mlflow.genai.scorers'

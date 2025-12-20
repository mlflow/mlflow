import os
from mlflow.genai.scorers import Correctness

scorer = Correctness()
scorer.register(experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID"))

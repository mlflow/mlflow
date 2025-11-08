"""
python examples/mlflow-3/langchain_databricks_example.py
"""

from databricks.sdk import WorkspaceClient
from langchain_core.runnables import RunnableLambda

import mlflow

mlflow.langchain.autolog(log_models=True)

wc = WorkspaceClient()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{wc.current_user.me().user_name}/langchain-autolog")

with mlflow.start_run() as run:
    r = RunnableLambda(lambda x: x + 1)
    r.invoke(3)

print(mlflow.search_traces(max_results=1))

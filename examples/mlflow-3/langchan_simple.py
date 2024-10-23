from langchain_core.runnables import RunnableLambda

import mlflow

mlflow.langchain.autolog()

with mlflow.start_run() as run:
    r = RunnableLambda(lambda x: x + 1)
    print(r.invoke(3))


print(mlflow.search_logged_models(experiment_ids=[run.info.experiment_id], output_format="list"))

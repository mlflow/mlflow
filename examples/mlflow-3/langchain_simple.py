import mlflow

mlflow.langchain.autolog(log_models=True)

from langchain_core.runnables import RunnableLambda

with mlflow.start_run() as run:
    r = RunnableLambda(lambda x: x + 1)
    r.invoke(3)


models = mlflow.search_logged_models(
    experiment_ids=[run.info.experiment_id],
    output_format="list",
)
loaded_model = mlflow.langchain.load_model(f"models:/{models[0].model_id}")
print(loaded_model.invoke(3))

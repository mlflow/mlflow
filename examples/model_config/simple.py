import mlflow

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model="model.py",
        model_config={"timeout": 10},
        input_example=["hello"],
    )


# model = mlflow.pyfunc.load_model(model_info.model_uri, model_config={"timeout": 10})
# print(model.predict("hello"))

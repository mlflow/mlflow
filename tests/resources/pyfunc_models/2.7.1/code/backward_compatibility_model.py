import mlflow


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input

import openai
import mlflow


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "system", "content": "You are an MLflow expert."}],
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict([{"role": "user", "content": "What is MLflow?"}]))

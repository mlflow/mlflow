import openai
import mlflow
import pandas as pd

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "system", "content": "You are an MLflow expert!"}],
    )

df = pd.DataFrame(
    {
        "role": ["user"] * 10,
        "content": [
            "What is MLflow?",
            "What are the key components of MLflow?",
            "How does MLflow enable reproducibility?",
            "What is MLflow tracking and how does it help?",
            "How can you compare different ML models using MLflow?",
            "How can you use MLflow to deploy ML models?",
            "What are the integrations of MLflow with popular ML libraries?",
            "How can you use MLflow to automate ML workflows?",
            "What security and compliance features does MLflow offer?",
            "Where does MLflow stand in the ML ecosystem?",
        ],
    }
)
model = mlflow.pyfunc.load_model(model_info.model_uri)
print(df.assign(answer=model.predict(df)))

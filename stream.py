import sys

import openai

import mlflow

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[
            {
                "role": "system",
                "content": "Please always answer in Japanese whatever language the user speaks",
            },
        ],
    )


model = mlflow.pyfunc.load_model(model_info.model_uri)
for c in model.predict_stream("How should I study machine learning?"):
    sys.stdout.write(c.choices[0].delta.content or "")

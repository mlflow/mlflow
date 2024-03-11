import subprocess
import sys
import time

import openai
import requests

import mlflow

with mlflow.start_run() as run:
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


print("-" * 10)
model = mlflow.pyfunc.load_model(model_info.model_uri)
for c in model.predict_stream("How should I study machine learning?"):
    sys.stdout.write(c.choices[0].delta.content or "")

print("-" * 10)


with subprocess.Popen(
    [
        sys.executable,
        "-m",
        "mlflow",
        "models",
        "serve",
        "-m",
        f"runs:/{run.info.run_id}/model",
        "--env-manager",
        "local",
    ],
) as proc:
    try:
        time.sleep(5)
        resp = requests.post(
            "http://localhost:5000/invocations",
            json={
                "inputs": ["Hello, how are you?"],
                "params": {
                    "stream": True,
                },
            },
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            print(line.decode("utf-8"))
    except requests.exceptions.HTTPError as err:
        print(err, resp.text)
    finally:
        proc.terminate()

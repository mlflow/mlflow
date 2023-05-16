import os
import sys
import mlflow
import datasets


import openai
import pandas as pd


def text_generation():
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="gpt-3.5-turbo",
            task=openai.ChatCompletion,
            artifact_path="model",
            messages=[{"role": "user", "content": "Tell me a funny joke about {animal}."}],
        )
        df = pd.DataFrame(
            {
                "animal": [
                    "cats",
                    "dogs",
                ],
                "target": [
                    "cats",
                    "dogs",
                ],
            }
        )
        mlflow.evaluate(
            model=model_info.model_uri,
            data=df,
            targets="target",
            model_type="text-generation",
        )


def question_answering():
    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="gpt-3.5-turbo",
            task=openai.ChatCompletion,
            artifact_path="model",
            messages=[
                {
                    "role": "system",
                    "content": "You're a calculator. Please only give me the answer, which means the response should only contain numbers.",
                },
                {"role": "user", "content": "{x} + {y} ="},
            ],
        )
        df = pd.DataFrame(
            {
                "x": ["1", "2"],
                "y": ["3", "4"],
                "target": ["4", "6"],
            }
        )
        mlflow.evaluate(
            model=model_info.model_uri,
            data=df,
            targets="target",
            model_type="question-answering",
        )


def text_summarization():
    dt = datasets.load_dataset("billsum", split="train[:10]")
    print(dt)

    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            model="gpt-3.5-turbo",
            task=openai.ChatCompletion,
            artifact_path="model",
            messages=[
                {"role": "user", "content": "Summarize the following text:\n\n---\n\n{text}"},
            ],
        )
        df = dt.to_pandas()
        mlflow.evaluate(
            model=model_info.model_uri,
            data=df,
            targets="summary",
            model_type="text-summarization",
        )


def information_retrieval():
    raise NotImplemented


if __name__ == "__main__":
    task = sys.argv[1]
    if task == "question-answering":
        question_answering()
    elif task == "text-summarization":
        text_summarization()
    elif task == "text-generation":
        text_generation()
    elif task == "retrieval":
        information_retrieval()
    else:
        raise ValueError(f"Invalid task: {task}")

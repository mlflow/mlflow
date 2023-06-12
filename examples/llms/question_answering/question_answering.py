import os
import pandas as pd

import mlflow

import openai


assert "OPENAI_API_KEY" in os.environ, (
    "Please set the OPENAI_API_KEY environment variable to run this example."
)

def evaluate_prompt(system_prompt):
    mlflow.start_run()
    mlflow.log_param("system_prompt", system_prompt)

    # Create a quesiton answering model using prompt engineering with OpenAI. Log the model
    # to MLflow Tracking
    logged_model = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"}
        ]
    )

    # Evaluate the model on some example questions
    questions = pd.DataFrame({
        "question": [
            "How do you create a run with MLflow?",
            "How do you log a model with MLflow?",
            "What is the capital of France?",
        ]
    })
    mlflow.evaluate(
        model=logged_model.model_uri,
        model_type="question-answering",
        data=questions,
    )
    mlflow.end_run()


system_prompt_1 = "Your job is to answer questions about MLflow."
evaluate_prompt(system_prompt_1)

system_prompt_2 = "Your job is to answer questions about MLflow. When you are asked a question about MLflow respond to it. Make sure to include code examples. If the question is not related to MLflow, refuse to answer and say that the question is unrelated."
evaluate_prompt(system_prompt_2)

# # Load the evaluation results
results: pd.DataFrame = mlflow.load_table("eval_results_table.json", extra_columns=["run_id", "params.system_prompt"])
results_grouped_by_question = results.sort_values(by="question")
print(results_grouped_by_question[["run_id", "params.system_prompt", "question", "outputs"]])

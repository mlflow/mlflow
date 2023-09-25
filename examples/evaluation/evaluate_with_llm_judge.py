import logging
import os

import openai
import pandas as pd

import mlflow
from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.utils.make_genai_metric import make_genai_metric

logging.getLogger("mlflow").setLevel(logging.ERROR)

# Uncomment the following lines to run this script without using a real OpenAI API key.
# os.environ["OPENAI_API_KEY"] = "test"

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# testing with OpenAI gpt-3.5-turbo
example = EvaluationExample(
    input="What is MLflow?",
    output="MLflow is an open-source platform for managing machine "
    "learning workflows, including experiment tracking, model packaging, "
    "versioning, and deployment, simplifying the ML lifecycle.",
    score=4,
    justification="The definition effectively explains what MLflow is "
    "its purpose, and its developer. It could be more concise for a 5-score.",
    variables={
        "ground_truth": "MLflow is an open-source platform for managing "
        "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
        "a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning "
        "engineers face when developing, training, and deploying machine learning models."
    },
)

correctness = make_genai_metric(
    name="correctness",
    version="v1",
    definition="Correctness refers to how well the generated output matches "
    "or aligns with the reference or ground truth text that is considered "
    "accurate and appropriate for the given input. The ground truth serves as "
    "a benchmark against which the provided output is compared to determine the "
    "level of accuracy and fidelity.",
    grading_prompt="Correctness: If the answer correctly answer the question, below are the "
    "details for different scores: "
    "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
    "the question or is completely contrary to the correct answer. "
    "- Score 1: the answer provides some relevance to the question and answer one aspect "
    "of the question correctly. "
    "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
    "critical aspect. "
    "- Score 4: the answer correctly answer the question and not missing any major aspect",
    examples=[example],
    model="openai:/gpt-3.5-turbo-16k",
    variables=["ground_truth"],
    parameters={"temperature": 0.0},
    greater_is_better=True,
    aggregations=["mean", "variance", "p90"],
)

eval_df = pd.DataFrame(
    {
        "input": [
            "What is MLflow?",
            "What is Spark?",
            "What is Python?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, offering improvements in speed and ease of use. Spark provides libraries for various tasks such as data ingestion, processing, and analysis through its components like Spark SQL for structured data, Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
            "Python is a high-level programming language that was created by Guido van Rossum and released in 1991. It emphasizes code readability and allows developers to express concepts in fewer lines of code than languages like C++ or Java. Python is used in various domains, including web development, scientific computing, data analysis, and machine learning.",
        ],
    }
)

with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    logged_model = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    results = mlflow.evaluate(
        logged_model.model_uri,
        eval_df,
        model_type="text",
        custom_metrics=[correctness],
    )
    print(results)

    eval_table = mlflow.load_table("eval_results_table.json", run_ids=[run.info.run_id])
    print(eval_table)

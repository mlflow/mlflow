import os
import pandas as pd

import mlflow

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

assert "OPENAI_API_KEY" in os.environ, (
    "Please set the OPENAI_API_KEY environment variable to run this example."
)

def evaluate_prompt(prompt_template):
    mlflow.start_run()
    mlflow.log_param("prompt_template", prompt_template)
    # Create a news summarization model using prompt engineering with LangChain. Log the model
    # to MLflow Tracking
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["article"],
        template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    logged_model = mlflow.langchain.log_model(chain, "langchain_model")

    # Evaluate the model on a small sample dataset
    sample_data = pd.read_csv("summarization_sample_data.csv")
    mlflow.evaluate(
        model = logged_model.model_uri,
        model_type="text-summarization",
        data=sample_data,
        targets="highlights",
    )
    mlflow.end_run()

prompt_template_1 = "Write a summary of the following article that is between triple backticks: ```{article}```"
evaluate_prompt(prompt_template_1)
prompt_template_2 = "Write a summary of the following article that is between triple backticks. Be concise. Make sure the summary includes important nouns and dates and keywords in the original text. Just return the summary. Do not include any text other than the summary: ```{article}```"
evaluate_prompt(prompt_template_2)

# Load the evaluation results
results: pd.DataFrame = mlflow.load_table("eval_results_table.json", extra_columns=["run_id", "params.prompt_template"])
results_grouped_by_article = results.sort_values(by="id")
print(results_grouped_by_article[["run_id", "params.prompt_template", "article", "outputs"]])

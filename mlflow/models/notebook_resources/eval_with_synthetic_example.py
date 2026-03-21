# ruff: noqa: F821, I001
{{pipInstall}}

from databricks.agents.evals import generate_evals_df
import mlflow

agent_description = "A chatbot that answers questions about Databricks."
question_guidelines = """
# User personas
- A developer new to the Databricks platform
# Example questions
- What API lets me parallelize operations over rows of a delta table?
"""
# TODO: Spark/Pandas DataFrame with "content" and "doc_uri" columns.
docs = spark.table("catalog.schema.my_table_of_docs")
evals = generate_evals_df(
    docs=docs,
    num_evals=25,
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)
eval_result = mlflow.evaluate(data=evals, model="{{modelUri}}", model_type="databricks-agent")

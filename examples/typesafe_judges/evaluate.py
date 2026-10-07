from typing import Literal

import mlflow
from mlflow.genai.judges import make_judge

MODEL = "typesafe:/jev-latest"

DATA = [
    {
        "inputs": {"question": "How do I reset my password?"},
        "outputs": "Select Forgot password on the sign-in page to receive a reset link.",
    },
    {
        "inputs": {"question": "Can I update the billing email for invoices?"},
        "outputs": "Yes. Open Billing settings and edit the invoice email address.",
    },
]


answer_helpfulness = make_judge(
    name="answer_helpfulness",
    instructions=(
        "Answer yes or no: Does {{ outputs }} directly answer the user's question in {{ inputs }}?"
    ),
    model=MODEL,
    feedback_value_type=bool,
)

request_category = make_judge(
    name="request_category",
    instructions="Which support category best describes the user's question in {{ inputs }}?",
    model=MODEL,
    feedback_value_type=Literal["account", "billing", "other"],
)


mlflow.set_experiment("typesafe-judge-example")

results = mlflow.genai.evaluate(
    data=DATA,
    scorers=[answer_helpfulness, request_category],
)

print(results.result_df[["answer_helpfulness/value", "request_category/value"]])

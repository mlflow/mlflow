import argparse

import mlflow
from mlflow.genai.scorers import make_jev_scorer

EVALUATION_DATA = [
    {
        "inputs": {"question": "How do I reset my password?"},
        "outputs": "Select Forgot password on the sign-in page to receive a reset link.",
        "expectations": {"expected_response": "Use the Forgot password link."},
    },
    {
        "inputs": {"question": "How do I download an invoice?"},
        "outputs": "Our office is open from 9 AM to 5 PM.",
        "expectations": {"expected_response": "Open Billing and select Download invoice."},
    },
]


def run_evaluation(model):
    mlflow.set_experiment("jev-evaluation")
    scorers = [
        make_jev_scorer(
            name="answer_relevance",
            model=model,
            question="Does outputs answer the question in inputs?",
            answer_type="noul",
            criteria={
                "true": "The answer directly addresses the user's question.",
                "false": "The answer is unrelated or does not address the question.",
            },
            threshold=0.7,
        ),
        make_jev_scorer(
            name="request_category",
            model=model,
            question="Which category describes the question in inputs?",
            answer_type="choice",
            criteria={
                "account": "Account access, sign-in, or password help.",
                "billing": "Invoices, payments, or subscriptions.",
                "other": "Another type of request.",
            },
        ),
        make_jev_scorer(
            name="answer_completeness",
            model=model,
            question="How completely does outputs answer inputs, considering expectations?",
            answer_type="score",
            criteria=[
                "Does not provide the requested information.",
                "Provides some useful information but misses a necessary step.",
                "Provides all the information needed to resolve the question.",
            ],
        ),
    ]
    return mlflow.genai.evaluate(data=EVALUATION_DATA, scorers=scorers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate support answers with TypeSafe Jev.")
    parser.add_argument(
        "--model",
        default="typesafe:/jev-latest",
        help="A typesafe:/ model URI or a gateway:/ endpoint URI.",
    )
    args = parser.parse_args()
    results = run_evaluation(args.model)
    print(f"Run ID: {results.run_id}")
    print(results.metrics)

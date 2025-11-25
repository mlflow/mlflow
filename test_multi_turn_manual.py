import os
import mlflow
from mlflow.genai import make_judge

@mlflow.trace(span_type="CHAT_MODEL")
def model(question, session_id):
    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    return f"Answer to: {question}"

mlflow.set_experiment("judge_test")
with mlflow.start_run() as run:
    for q in ["What is ML?", "How does it work?", "Example?"]:
        model(q, session_id="conv_1")

    traces = mlflow.search_traces(
        experiment_ids=[run.info.experiment_id],
        filter_string=f'run_id = "{run.info.run_id}"'
    )

    # Using {{ conversation }} makes it session-level automatically
    judge = make_judge(
        name="coherence",
        model="openai:/gpt-4o-mini",
        instructions="Evaluate conversation coherence: {{ conversation }}. Return True if coherent.",
        feedback_value_type=bool
    )

    print(f"Is session-level: {judge.is_session_level_scorer}")

    results = mlflow.genai.evaluate(data=traces, scorers=[judge])
    assessments = results.result_df["coherence/value"].notna().sum()
    print(f"Assessments: {assessments} (expected: 1)")

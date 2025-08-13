import pytest
from dspy import Predict

pytest.importorskip(
    "dspy", minversion="2.6.12", reason="evaluate callback is available since 2.6.12"
)

import dspy
from dspy import Evaluate, Example
from dspy.evaluate.metrics import answer_exact_match
from dspy.utils.dummies import DummyLM

import mlflow
from mlflow.genai.optimize.optimizers.utils.dspy_mipro_callback import _DSPyMIPROv2Callback


@pytest.fixture
def callback():
    return _DSPyMIPROv2Callback(
        prompt_name="test_prompt",
        input_fields={"question": str, "context": str},
        convert_to_single_text=True,
    )


def test_callback_with_evals(callback):
    class EvalOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, eval, trainset, valset):
            eval(program, devset=valset, callback_metadata={"metric_key": "eval_full"})
            eval(program, devset=valset[:1], callback_metadata={"metric_key": "eval_full"})
            return program

    lm = DummyLM(
        {
            "What is 1 + 1?": {"answer": "2"},
            "What is 2 + 2?": {"answer": "1000"},
        }
    )

    dataset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]
    program = Predict("question -> answer")
    evaluator = Evaluate(devset=dataset, metric=answer_exact_match)
    optimizer = EvalOptimizer()

    with dspy.context(
        lm=lm,
        callbacks=[callback],
        adapter=dspy.ChatAdapter(),
    ):
        with mlflow.start_run():
            optimizer.compile(program, evaluator, trainset=dataset, valset=dataset)

    # root run
    run = mlflow.last_active_run()
    assert run.data.metrics == {
        "eval_full": 100.0,
    }
    prompt = mlflow.load_prompt("prompts:/test_prompt/1")
    assert prompt.tags["overall_eval_score"] == "100.0"

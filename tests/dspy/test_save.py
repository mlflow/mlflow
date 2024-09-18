import json

import dspy
import dspy.teleprompt
import pytest

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from tests.helper_functions import expect_status_code, pyfunc_serve_and_score_model


@pytest.fixture
def cleanup_fixture():
    yield

    dspy.settings.configure(lm=None, rm=None)


def test_basic_save(cleanup_fixture):
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    dspy_model = CoT()
    dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini", max_tokens=250))

    with mlflow.start_run() as run:
        mlflow.dspy.log_model(dspy_model, "model")

    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    model_path = "model"
    model_url = f"runs:/{run.info.run_id}/{model_path}"
    loaded_model = mlflow.dspy.load_model(model_url)

    # Check that the global settings is popped back.
    assert dspy.settings.lm.kwargs["model"] == "gpt-4o-mini"
    assert isinstance(loaded_model, CoT)


def test_save_compiled_model(cleanup_fixture):
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    train_data = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?", "What is 5 + 5?"]
    train_label = ["4", "6", "8", "10"]
    trainset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in zip(train_data, train_label)
    ]

    def dummy_metric(program):
        return 1.0

    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    dspy_model = CoT()
    optimizer = dspy.teleprompt.BootstrapFewShot(metric=dummy_metric)
    optimized_cot = optimizer.compile(dspy_model, trainset=trainset)

    with mlflow.start_run() as run:
        mlflow.dspy.log_model(optimized_cot, "model")

    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    model_path = "model"
    model_url = f"runs:/{run.info.run_id}/{model_path}"
    loaded_model = mlflow.dspy.load_model(model_url)

    assert isinstance(loaded_model, CoT)
    assert loaded_model.prog.predictors()[0].demos == optimized_cot.prog.predictors()[0].demos


def test_save_model_with_multiple_modules(cleanup_fixture):
    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factoid answers."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):
        def __init__(self, num_passages=3):
            super().__init__()

            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

    def dummy_metric(program):
        return 1.0

    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DummyLM(answers=random_answers)
    rm = dspy.utils.dummy_rm(passages=["dummy1", "dummy2", "dummy3"])
    dspy.settings.configure(lm=lm, rm=rm)

    train_data = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?", "What is 5 + 5?"]
    train_label = ["4", "6", "8", "10"]
    trainset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in zip(train_data, train_label)
    ]

    dspy_model = RAG()
    optimizer = dspy.teleprompt.BootstrapFewShot(metric=dummy_metric)
    optimized_cot = optimizer.compile(dspy_model, trainset=trainset)

    with mlflow.start_run() as run:
        mlflow.dspy.log_model(optimized_cot, "model")

    original_settings = dict(dspy.settings.config)
    original_settings["traces"] = None

    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    model_path = "model"
    model_url = f"runs:/{run.info.run_id}/{model_path}"

    input_examples = {"inputs": ["What is 2 + 2?"]}
    # test that the model can be served
    response = pyfunc_serve_and_score_model(
        model_uri=model_url,
        data=json.dumps(input_examples),
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )
    expect_status_code(response, 200)

    loaded_model = mlflow.dspy.load_model(model_url)
    assert isinstance(loaded_model, RAG)
    assert loaded_model.retrieve is not None
    assert (
        loaded_model.generate_answer.predictors()[0].demos
        == optimized_cot.generate_answer.predictors()[0].demos
    )

    loaded_settings = dict(dspy.settings.config)
    loaded_settings["traces"] = None

    assert loaded_settings["lm"].__dict__ == original_settings["lm"].__dict__
    assert loaded_settings["rm"].__dict__ == original_settings["rm"].__dict__

    del (
        loaded_settings["lm"],
        original_settings["lm"],
        loaded_settings["rm"],
        original_settings["rm"],
    )

    assert original_settings == loaded_settings


def test_serving_logged_model(cleanup_fixture):
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    dspy_model = CoT()
    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    input_examples = {"inputs": ["What is 2 + 2?"]}
    input_schema = Schema([ColSpec("string")])
    output_schema = Schema([ColSpec("string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run() as run:
        mlflow.dspy.log_model(
            dspy_model, "model", input_example=input_examples, signature=signature
        )

    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    model_path = "model"
    model_url = f"runs:/{run.info.run_id}/{model_path}"

    # test that the model can be served
    response = pyfunc_serve_and_score_model(
        model_uri=model_url,
        data=json.dumps(input_examples),
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)

    json_response = json.loads(response.content)
    # Assert the required fields are in the response.
    assert "rationale" in json_response["predictions"]
    assert "answer" in json_response["predictions"]

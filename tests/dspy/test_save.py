import json
from unittest import mock

import dspy
import dspy.teleprompt
import pytest

import mlflow
from mlflow.models import Model, ModelSignature
from mlflow.types.schema import ColSpec, Schema

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
    expect_status_code,
    pyfunc_serve_and_score_model,
)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    yield

    dspy.settings.configure(lm=None, rm=None)


def test_basic_save():
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


def test_save_compiled_model():
    train_data = ["What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?", "What is 5 + 5?"]
    train_label = ["4", "6", "8", "10"]
    trainset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in zip(train_data, train_label)
    ]

    def dummy_metric(program):
        return 1.0

    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
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


def test_dspy_save_preserves_object_state():
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
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
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


def test_load_logged_model_in_native_dspy():
    dspy_model = CoT()
    # Arbitrary set the demo to test saving/loading has no data loss.
    dspy_model.prog.predictors()[0].demos = [
        "What is 2 + 2?",
        "What is 3 + 3?",
        "What is 4 + 4?",
        "What is 5 + 5?",
    ]
    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    with mlflow.start_run() as run:
        mlflow.dspy.log_model(dspy_model, "model")
    model_path = "model"
    model_url = f"runs:/{run.info.run_id}/{model_path}"
    loaded_dspy_model = mlflow.dspy.load_model(model_url)

    assert isinstance(loaded_dspy_model, CoT)
    assert loaded_dspy_model.prog.predictors()[0].demos == dspy_model.prog.predictors()[0].demos


def test_serving_logged_model():
    # Need to redefine a CoT in the test case for cloudpickle to find the class.
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    dspy_model = CoT()
    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    input_examples = {"inputs": ["What is 2 + 2?"]}
    input_schema = Schema([ColSpec("string")])
    output_schema = Schema([ColSpec("string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.dspy.log_model(
            dspy_model,
            artifact_path,
            signature=signature,
            input_example=input_examples,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    # test that the model can be served
    response = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=json.dumps(input_examples),
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)

    json_response = json.loads(response.content)

    # Assert the required fields are in the response.
    assert "rationale" in json_response["predictions"]
    assert "answer" in json_response["predictions"]


def test_save_chat_model_with_string_output():
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, inputs):
            # DSPy chat model's inputs is a list of dict with keys roles (optional) and content.
            # And here we output a single string.
            return self.prog(question=inputs[0]["content"]).answer

    dspy_model = CoT()
    random_answers = ["4", "4", "4", "4"]
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    input_examples = {"messages": [{"role": "user", "content": "What is 2 + 2?"}]}

    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.dspy.log_model(
            dspy_model,
            artifact_path,
            task="llm/v1/chat",
            input_example=input_examples,
        )
    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    response = loaded_pyfunc.predict(input_examples)

    assert "choices" in response
    assert len(response["choices"]) == 1
    assert "message" in response["choices"][0]
    # The content should just be a string.
    assert response["choices"][0]["message"]["content"] == "4"


def test_serve_chat_model():
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, inputs):
            # DSPy chat model's inputs is a list of dict with keys roles (optional) and content.
            return self.prog(question=inputs[0]["content"])

    dspy_model = CoT()
    random_answers = ["4", "6", "8", "10"]
    lm = dspy.utils.DSPDummyLM(answers=random_answers)
    dspy.settings.configure(lm=lm)

    input_examples = {"messages": [{"role": "user", "content": "What is 2 + 2?"}]}

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.dspy.log_model(
            dspy_model,
            artifact_path,
            task="llm/v1/chat",
            input_example=input_examples,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    # Clear the lm setting to test the loading logic.
    dspy.settings.configure(lm=None)

    # test that the model can be served
    response = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=json.dumps(input_examples),
        content_type="application/json",
        extra_args=["--env-manager", "local"],
    )

    expect_status_code(response, 200)

    json_response = json.loads(response.content)

    assert "choices" in json_response
    assert len(json_response["choices"]) == 1
    assert "message" in json_response["choices"][0]
    assert "rationale" in json_response["choices"][0]["message"]["content"]
    assert "answer" in json_response["choices"][0]["message"]["content"]


def test_code_paths_is_used():
    artifact_path = "model"
    dspy_model = CoT()
    with mlflow.start_run(), mock.patch(
        "mlflow.dspy.load._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.dspy.log_model(dspy_model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, "dspy")
        mlflow.dspy.load_model(model_uri)
        add_mock.assert_called()


def test_additional_pip_requirements():
    expected_mlflow_version = _mlflow_major_version_string()
    artifact_path = "model"
    dspy_model = CoT()
    with mlflow.start_run():
        mlflow.dspy.log_model(dspy_model, artifact_path, extra_pip_requirements=["dummy"])

        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "dummy"]
        )


def test_infer_signature_from_input_examples():
    artifact_path = "model"
    dspy_model = CoT()
    random_answers = ["4", "6", "8", "10"]
    dspy.settings.configure(lm=dspy.utils.DSPDummyLM(answers=random_answers))
    with mlflow.start_run():
        mlflow.dspy.log_model(dspy_model, artifact_path, input_example="what is 2 + 2?")

        model_uri = mlflow.get_artifact_uri(artifact_path)
        loaded_model = Model.load(model_uri)
        assert loaded_model.signature.inputs == Schema([ColSpec("string")])
        assert loaded_model.signature.outputs == Schema(
            [ColSpec(name="rationale", type="string"), ColSpec(name="answer", type="string")]
        )

import importlib.metadata
import json

import dspy
import pytest
from dspy.teleprompt import LabeledFewShot
from packaging.version import Version
from pydantic import HttpUrl

import mlflow
import mlflow.dspy.util
from mlflow.dspy.util import (
    log_dspy_dataset,
    log_dspy_module_params,
    save_dspy_module_state,
)
from mlflow.tracking import MlflowClient


@pytest.mark.skipif(
    Version(importlib.metadata.version("dspy")) < Version("2.5.43"),
    reason="dump_state works differently in older versions",
)
def test_save_dspy_module_state(tmp_path):
    program = dspy.ChainOfThought("question -> answer")

    with mlflow.start_run() as run:
        save_dspy_module_state(program)

    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "model.json" in artifacts
    client.download_artifacts(run_id=run.info.run_id, path="model.json", dst_path=tmp_path)
    loaded_program = dspy.ChainOfThought("b -> a")
    loaded_program.load(tmp_path / "model.json")
    assert loaded_program.dump_state() == program.dump_state()


def test_log_dspy_module_state_params():
    program = dspy.Predict("question -> answer: list[str]")
    program.demos = [
        dspy.Example(question="What are cities in Japan?", answer=["Tokyo", "Osaka"]).with_inputs(
            "question"
        ),
    ]

    with mlflow.start_run() as run:
        log_dspy_module_params(program)

    run = mlflow.last_active_run()
    assert run.data.params == {
        "Predict.signature.fields.0.description": "${question}",
        "Predict.signature.fields.0.prefix": "Question:",
        "Predict.signature.fields.1.description": "${answer}",
        "Predict.signature.fields.1.prefix": "Answer:",
        "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
        "Predict.demos.0.answer": "['Tokyo', 'Osaka']",
        "Predict.demos.0.question": "What are cities in Japan?",
    }


def test_log_dataset(tmp_path):
    dataset = [
        dspy.Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    with mlflow.start_run() as run:
        log_dspy_dataset(dataset, "dataset.json")

    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "dataset.json" in artifacts
    client.download_artifacts(run_id=run.info.run_id, path="dataset.json", dst_path=tmp_path)
    saved_dataset = json.loads((tmp_path / "dataset.json").read_text())
    assert saved_dataset == {
        "columns": ["question", "answer"],
        "data": [
            ["What is 1 + 1?", "2"],
            ["What is 2 + 2?", "4"],
        ],
    }


@pytest.mark.skipif(
    Version(importlib.metadata.version("dspy")) < Version("2.5.43"),
    reason="dump_state works differently in older versions",
)
@pytest.mark.parametrize(
    ("test_scenario", "file_name", "expected_exception", "expected_message_pattern"),
    [
        # Test with real DSPy HttpUrl objects (original issue scenario)
        (
            "httpurl_objects",
            "model.json",
            RuntimeError,
            (
                "JSON serialization failed[\\s\\S]*To resolve this, use: "
                "mlflow.dspy.autolog\\(save_program_with_pickle=True\\)"
            ),
        ),
        # Test with mock JSON serialization failure
        (
            "mock_json_failure",
            "model.json",
            RuntimeError,
            (
                "JSON serialization failed[\\s\\S]*To resolve this, use: "
                "mlflow.dspy.autolog\\(save_program_with_pickle=True\\)"
            ),
        ),
        # Test non-.json files raise original error
        ("non_json_file", "model.pkl", ValueError, "Some other error not related to JSON"),
    ],
)
def test_save_dspy_module_state_error_handling(
    test_scenario, file_name, expected_exception, expected_message_pattern
):
    """Test save_dspy_module_state error handling for different scenarios."""

    if test_scenario == "httpurl_objects":
        # Create a DSPy signature with non-JSON serializable field (HttpUrl)
        class TestSignature(dspy.Signature):
            query: str = dspy.InputField()
            url: HttpUrl = dspy.InputField(desc="A URL field that causes serialization issues")
            response: str = dspy.OutputField()

        class TestModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict(TestSignature)

            def forward(self, query, url):
                return self.predictor(query=query, url=url)

        # Create training data with non-JSON-serializable objects (HttpUrl)
        trainset = [
            dspy.Example(
                query="test query", url=HttpUrl("https://example.com"), response="test response"
            ).with_inputs("query", "url"),
        ]

        # Compile the program (this adds the trainset to demos, making it non-JSON serializable)
        program = TestModule()
        teleprompter = LabeledFewShot()
        test_program = teleprompter.compile(program, trainset=trainset)

    elif test_scenario == "mock_json_failure":
        # Create a mock program that fails JSON serialization
        class FailingJSONProgram:
            def save(self, path, save_program=False):
                if save_program:
                    # Would succeed with pickle
                    path.mkdir(exist_ok=True)
                    (path / "program.pkl").write_bytes(b"mock pickle data")
                else:
                    # Simulate JSON serialization failure
                    raise TypeError("Object of type SomeCustomType is not JSON serializable")

            def dump_state(self):
                return {"test": "data"}

        test_program = FailingJSONProgram()

    elif test_scenario == "non_json_file":
        # Create a program that fails for non-JSON reasons
        class FailingProgram:
            def save(self, path, save_program=False):
                # Unused parameters are expected in this mock
                raise ValueError("Some other error not related to JSON")

            def dump_state(self):
                return {"test": "data"}

        test_program = FailingProgram()

    # Test the error handling
    with mlflow.start_run():
        with pytest.raises(expected_exception, match=expected_message_pattern):
            save_dspy_module_state(test_program, file_name)


def test_save_dspy_module_state_force_pickle(tmp_path):
    """Test that save_dspy_module_state respects the force_pickle parameter."""

    class TestProgram:
        def save(self, path, save_program=False):
            if save_program:
                # Simulate successful pickle save - DSPy saves to directory
                path.mkdir(exist_ok=True)
                with open(path / "program.pkl", "wb") as f:
                    f.write(b"pickle data with save_program=True")
            else:
                # Simulate successful JSON save
                with open(path, "w") as f:
                    f.write('{"json": "data"}')

        def dump_state(self):
            return {"test": "data"}

    program = TestProgram()

    # Test force_pickle=True
    with mlflow.start_run() as run:
        save_dspy_module_state(program, "model.pkl", force_pickle=True)

    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run.info.run_id)]

    # Should have used pickle serialization (saved as directory without suffix)
    assert "model" in artifacts

    # Download and verify the directory was created with save_program=True
    client.download_artifacts(run_id=run.info.run_id, path="model", dst_path=tmp_path)
    assert (
        tmp_path / "model" / "program.pkl"
    ).read_bytes() == b"pickle data with save_program=True"

    # Test force_pickle=False (default behavior)
    with mlflow.start_run() as run:
        save_dspy_module_state(program, "model.json", force_pickle=False)

    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run.info.run_id)]

    # Should have used JSON serialization
    assert "model.json" in artifacts

    # Download and verify the JSON file was created without save_program
    client.download_artifacts(run_id=run.info.run_id, path="model.json", dst_path=tmp_path)
    assert (tmp_path / "model.json").read_text() == '{"json": "data"}'


@pytest.mark.parametrize(
    ("error_type", "original_exception", "original_message"),
    [
        ("standard_json", TypeError, "Object of type HttpUrl is not JSON serializable"),
        ("ujson_style", TypeError, "keys must be a string"),
        ("value_error", ValueError, "Cannot serialize this object to JSON"),
        ("runtime_error", RuntimeError, "Serialization error occurred"),
    ],
    ids=["standard_json", "ujson_style", "value_error", "runtime_error"],
)
def test_json_error_detection_robustness(error_type, original_exception, original_message):
    """Test various JSON serialization errors are properly detected and raise informative errors."""

    # Mock program that raises different types of JSON errors
    class FailingProgram:
        def save(self, path, save_program=False):
            if save_program:
                # Would succeed with pickle
                path.mkdir(exist_ok=True)
                (path / "program.pkl").write_bytes(b"pickle data")
            else:
                # Raise the specified error type for JSON attempts
                raise original_exception(original_message)

        def dump_state(self):
            return {"test": "data"}

    program = FailingProgram()

    with mlflow.start_run():
        # Should raise informative error for all JSON-related failures
        with pytest.raises(
            RuntimeError,
            match=(
                "JSON serialization failed[\\s\\S]*To resolve this, use: "
                "mlflow.dspy.autolog\\(save_program_with_pickle=True\\)"
            ),
        ):
            save_dspy_module_state(program, "model.json")

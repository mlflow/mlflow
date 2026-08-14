import importlib.metadata
import json

import dspy
import pytest
from packaging.version import Version

import mlflow
from mlflow.dspy.util import (
    log_dspy_dataset,
    log_dspy_lm_state,
    log_dspy_module_params,
    sanitize_params,
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

    expected_params = {
        "Predict.signature.fields.0.description": "${question}",
        "Predict.signature.fields.0.prefix": "Question:",
        "Predict.signature.fields.1.description": "${answer}",
        "Predict.signature.fields.1.prefix": "Answer:",
        "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
        "Predict.demos.0.question": "What are cities in Japan?",
    }

    # `log_dspy_module_params` stringifies the fields of demos serialized as `dspy.Example`
    # objects but flattens list values of demos serialized as plain dicts. Which one
    # `dump_state` emits depends on the DSPy version (e.g. 3.0.0 still uses `Example`, later
    # 3.x releases use plain dicts), so derive the expectation from the actual serialization.
    demo_state = (program.dump_state().get("demos") or [None])[0]
    if isinstance(demo_state, dspy.Example):
        expected_params["Predict.demos.0.answer"] = "['Tokyo', 'Osaka']"
    else:
        expected_params.update({
            "Predict.demos.0.answer.0": "Tokyo",
            "Predict.demos.0.answer.1": "Osaka",
        })

    assert run.data.params == expected_params


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


def test_log_dspy_lm_state():
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        cache=True,
        top_p=0.9,
        api_key="secret-key",
        api_base="https://api.openai.com",
    )
    with dspy.context(lm=lm):
        with mlflow.start_run():
            log_dspy_lm_state()

        run = mlflow.last_active_run()
        assert "lm_params" in run.data.params

        lm_params = json.loads(run.data.params["lm_params"])

        # Verify expected attributes are present
        assert lm_params == {
            "model": "openai/gpt-4o-mini",
            "cache": True,
            "model_type": "chat",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        }

        # Verify sensitive attributes are filtered out
        assert "api_key" not in lm_params
        assert "api_base" not in lm_params


def test_sanitize_params():
    params = {
        "api_key": "secret-key",
        "api_base": "https://api.openai.com",
        "azure_ad_token": "secret-token",
        "client_secret": "secret-client-secret",
        "azure_password": "secret-azure-password",
        "model": "gpt-4o-mini",
    }
    assert sanitize_params(params) == {"model": "gpt-4o-mini"}

    assert sanitize_params({}) == {}

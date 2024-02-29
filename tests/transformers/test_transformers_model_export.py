import base64
import gc
import importlib.util
import json
import os
import pathlib
import re
import shutil
import textwrap
from pathlib import Path
from unittest import mock

import huggingface_hub
import librosa
import numpy as np
import pandas as pd
import pytest
import torch
import transformers
import yaml
from datasets import load_dataset
from huggingface_hub import ModelCard
from packaging.version import Version

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature, infer_signature
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.transformers import (
    _CARD_DATA_FILE_NAME,
    _CARD_TEXT_FILE_NAME,
    _build_pipeline_from_model_input,
    _fetch_model_card,
    _is_model_distributed_in_memory,
    _should_add_pyfunc_to_model,
    _TransformersWrapper,
    _validate_llm_inference_task_type,
    _write_card_data,
    _write_license_information,
    get_default_conda_env,
    get_default_pip_requirements,
)
from mlflow.types.schema import Array, ColSpec, DataType, ParamSchema, ParamSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _compare_logged_code_paths,
    _get_deps_from_requirement_file,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    pyfunc_serve_and_score_model,
)
from tests.transformers.helper import IS_NEW_FEATURE_EXTRACTION_API, flaky
from tests.transformers.test_transformers_peft_model import SKIP_IF_PEFT_NOT_AVAILABLE

_IS_PIPELINE_DTYPE_SUPPORTED_VERSION = Version(transformers.__version__) >= Version("4.26.1")

# NB: Some pipelines under test in this suite come very close or outright exceed the
# default runner containers specs of 7GB RAM. Due to this inability to run the suite without
# generating a SIGTERM Error (143), some tests are marked as local only.
# See: https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted- \
# runners#supported-runners-and-hardware-resources for instance specs.
RUNNING_IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
GITHUB_ACTIONS_SKIP_REASON = "Test consumes too much memory"
image_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/cat.png"
image_file_path = pathlib.Path(pathlib.Path(__file__).parent.parent, "datasets", "cat.png")
# Test that can only be run locally:
# - Summarization pipeline tests
# - TextClassifier pipeline tests
# - Text2TextGeneration pipeline tests
# - Conversational pipeline tests


@pytest.fixture(autouse=True)
def force_gc():
    # This reduces the memory pressure for the usage of the larger pipeline fixtures ~500MB - 1GB
    gc.disable()
    gc.collect()
    gc.set_threshold(0)
    gc.collect()
    gc.enable()


@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")
    yield model_path

    # Pytest keeps the temporary directory created by `tmp_path` fixture for 3 recent test sessions
    # by default. This is useful for debugging during local testing, but in CI it just wastes the
    # disk space.
    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture
def transformers_custom_env(tmp_path):
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["transformers"])
    return conda_env


@pytest.fixture
def mock_pyfunc_wrapper():
    return mlflow.transformers._TransformersWrapper("mock")


@pytest.fixture
@flaky()
def image_for_test():
    dataset = load_dataset("huggingface/cats-image")
    return dataset["test"]["image"][0]


@pytest.fixture
def sound_file_for_test():
    datasets_path = pathlib.Path(__file__).resolve().parent.parent.joinpath("datasets")
    audio, _ = librosa.load(datasets_path.joinpath("apollo11_launch.wav"), sr=16000)
    return audio


@pytest.fixture
def raw_audio_file():
    return read_raw_audio_file()


def read_raw_audio_file():
    datasets_path = pathlib.Path(__file__).resolve().parent.parent.joinpath("datasets")

    return datasets_path.joinpath("apollo11_launch.wav").read_bytes()


@pytest.mark.parametrize(
    ("pipeline", "expected_requirements"),
    [
        ("small_qa_pipeline", {"transformers", "torch", "torchvision"}),
        ("small_seq2seq_pipeline", {"transformers", "tensorflow"}),
        pytest.param(
            "peft_pipeline",
            {"peft", "transformers", "torch", "torchvision"},
            marks=SKIP_IF_PEFT_NOT_AVAILABLE,
        ),
    ],
)
def test_default_requirements(pipeline, expected_requirements, request):
    if "torch" in expected_requirements and importlib.util.find_spec("accelerate"):
        expected_requirements.add("accelerate")

    model = request.getfixturevalue(pipeline).model
    pip_requirements = get_default_pip_requirements(model)
    conda_requirements = get_default_conda_env(model)["dependencies"][2]["pip"]

    def _strip_requirements(requirements):
        return {req.split("==")[0] for req in requirements}

    assert _strip_requirements(pip_requirements) == expected_requirements
    assert _strip_requirements(conda_requirements) == (expected_requirements | {"mlflow"})


def test_inference_task_validation(small_seq2seq_pipeline, text_generation_pipeline):
    with pytest.raises(
        MlflowException, match="The task provided is invalid. 'llm/v1/invalid' is not"
    ):
        _validate_llm_inference_task_type("llm/v1/invalid", text_generation_pipeline)
    with pytest.raises(
        MlflowException, match="The task provided is invalid. 'llm/v1/completions' is not"
    ):
        _validate_llm_inference_task_type("llm/v1/completions", small_seq2seq_pipeline)
    _validate_llm_inference_task_type("llm/v1/completions", text_generation_pipeline)


@pytest.mark.parametrize(
    ("model", "result"),
    [
        ("small_qa_pipeline", True),
        ("small_seq2seq_pipeline", True),
        ("small_multi_modal_pipeline", False),
        ("small_vision_model", True),
    ],
)
def test_pipeline_eligibility_for_pyfunc_registration(model, result, request):
    pipeline = request.getfixturevalue(model)
    assert _should_add_pyfunc_to_model(pipeline) == result


def test_component_multi_modal_model_ineligible_for_pyfunc(component_multi_modal):
    task = transformers.pipelines.get_task(component_multi_modal["model"].name_or_path)
    pipeline = _build_pipeline_from_model_input(component_multi_modal, task)
    assert not _should_add_pyfunc_to_model(pipeline)


def test_pipeline_construction_from_base_nlp_model(small_qa_pipeline):
    generated = _build_pipeline_from_model_input(
        {"model": small_qa_pipeline.model, "tokenizer": small_qa_pipeline.tokenizer},
        "question-answering",
    )
    assert isinstance(generated, type(small_qa_pipeline))
    assert isinstance(generated.tokenizer, type(small_qa_pipeline.tokenizer))


def test_pipeline_construction_from_base_vision_model(small_vision_model):
    model = {"model": small_vision_model.model, "tokenizer": small_vision_model.tokenizer}
    if IS_NEW_FEATURE_EXTRACTION_API:
        model.update({"image_processor": small_vision_model.feature_extractor})
    else:
        model.update({"feature_extractor": small_vision_model.feature_extractor})
    generated = _build_pipeline_from_model_input(model, task="image-classification")
    assert isinstance(generated, type(small_vision_model))
    assert isinstance(generated.tokenizer, type(small_vision_model.tokenizer))
    if IS_NEW_FEATURE_EXTRACTION_API:
        compare_type = generated.image_processor
    else:
        compare_type = generated.feature_extractor
    assert isinstance(compare_type, transformers.MobileNetV2ImageProcessor)


def test_saving_with_invalid_dict_as_model(model_path):
    with pytest.raises(
        MlflowException, match="Invalid dictionary submitted for 'transformers_model'. The "
    ):
        mlflow.transformers.save_model(transformers_model={"invalid": "key"}, path=model_path)

    with pytest.raises(
        MlflowException, match="The 'transformers_model' dictionary must have an entry"
    ):
        mlflow.transformers.save_model(
            transformers_model={"tokenizer": "some_tokenizer"}, path=model_path
        )


def test_model_card_acquisition_vision_model(small_vision_model):
    model_provided_card = _fetch_model_card(small_vision_model.model.name_or_path)
    assert model_provided_card.data.to_dict()["tags"] == ["vision", "image-classification"]
    assert len(model_provided_card.text) > 0


@pytest.mark.parametrize(
    ("repo_id", "license_file"),
    [
        ("google/mobilenet_v2_1.0_224", "LICENSE.txt"),  # no license declared
        ("csarron/mobilebert-uncased-squad-v2", "LICENSE.txt"),  # mit license
        ("codellama/CodeLlama-34b-hf", "LICENSE"),  # custom license
        ("openai/whisper-tiny", "LICENSE.txt"),  # apache license
        ("stabilityai/stable-code-3b", "LICENSE"),  # custom
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "LICENSE.txt"),  # apache
    ],
)
def test_license_acquisition(repo_id, license_file, tmp_path):
    card_data = _fetch_model_card(repo_id)
    _write_license_information(repo_id, card_data, tmp_path)
    assert tmp_path.joinpath(license_file).stat().st_size > 0


def test_license_fallback(tmp_path):
    _write_license_information("not a real repo", None, tmp_path)
    assert tmp_path.joinpath("LICENSE.txt").stat().st_size > 0


def test_vision_model_save_pipeline_with_defaults(small_vision_model, model_path):
    mlflow.transformers.save_model(transformers_model=small_vision_model, path=model_path)
    # validate inferred pip requirements
    requirements = model_path.joinpath("requirements.txt").read_text()
    reqs = {req.split("==")[0] for req in requirements.split("\n")}
    expected_requirements = {"torch", "torchvision", "transformers"}
    assert reqs.intersection(expected_requirements) == expected_requirements
    # validate inferred model card data
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["tags"] == ["vision", "image-classification"]
    # verify the license file has been written
    license_file = model_path.joinpath("LICENSE.txt").read_text()
    assert len(license_file) > 0
    # Validate inferred model card text
    with model_path.joinpath("model_card.md").open() as file:
        card_text = file.read()
    assert len(card_text) > 0
    # Validate conda.yaml
    conda_env = yaml.safe_load(model_path.joinpath("conda.yaml").read_bytes())
    assert {req.split("==")[0] for req in conda_env["dependencies"][2]["pip"]}.intersection(
        expected_requirements
    ) == expected_requirements
    # Validate the MLModel file
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "ImageClassificationPipeline"
    assert flavor_config["pipeline_model_type"] == "MobileNetV2ForImageClassification"
    assert flavor_config["task"] == "image-classification"
    assert flavor_config["source_model_name"] == "google/mobilenet_v2_1.0_224"


def test_vision_model_save_model_for_task_and_card_inference(small_vision_model, model_path):
    mlflow.transformers.save_model(transformers_model=small_vision_model, path=model_path)
    # validate inferred pip requirements
    requirements = model_path.joinpath("requirements.txt").read_text()
    reqs = {req.split("==")[0] for req in requirements.split("\n")}
    expected_requirements = {"torch", "torchvision", "transformers"}
    assert reqs.intersection(expected_requirements) == expected_requirements
    # validate inferred model card data
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["tags"] == ["vision", "image-classification"]
    # Validate inferred model card text
    card_text = model_path.joinpath("model_card.md").read_text(encoding="utf-8")
    assert len(card_text) > 0
    # verify the license file has been written
    license_file = model_path.joinpath("LICENSE.txt").read_text()
    assert len(license_file) > 0
    # Validate the MLModel file
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "ImageClassificationPipeline"
    assert flavor_config["pipeline_model_type"] == "MobileNetV2ForImageClassification"
    assert flavor_config["task"] == "image-classification"
    assert flavor_config["source_model_name"] == "google/mobilenet_v2_1.0_224"


def test_qa_model_save_model_for_task_and_card_inference(small_seq2seq_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model={
            "model": small_seq2seq_pipeline.model,
            "tokenizer": small_seq2seq_pipeline.tokenizer,
        },
        path=model_path,
    )
    # validate inferred pip requirements
    with model_path.joinpath("requirements.txt").open() as file:
        requirements = file.read()
    reqs = {req.split("==")[0] for req in requirements.split("\n")}
    expected_requirements = {"tensorflow", "transformers"}
    assert reqs.intersection(expected_requirements) == expected_requirements
    # validate that the card was acquired by model reference
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["datasets"] == ["emo"]
    # The creator of this model did not include tag data in the card. Ensure it is missing.
    assert "tags" not in card_data
    # verify the license file has been written
    license_file = model_path.joinpath("LICENSE.txt").read_text()
    assert len(license_file) > 0
    # Validate inferred model card text
    with model_path.joinpath("model_card.md").open() as file:
        card_text = file.read()
    assert len(card_text) > 0
    # validate MLmodel files
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "TextClassificationPipeline"
    assert flavor_config["pipeline_model_type"] == "TFMobileBertForSequenceClassification"
    assert flavor_config["task"] == "text-classification"
    assert flavor_config["source_model_name"] == "lordtt13/emo-mobilebert"


def test_qa_model_save_and_override_card(small_qa_pipeline, model_path):
    supplied_card = """
                    ---
                    language: en
                    license: bsd
                    ---

                    # I made a new model!
                    """
    card_info = textwrap.dedent(supplied_card)
    card = ModelCard(card_info)
    # save the model instance
    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=model_path,
        model_card=card,
    )
    # validate that the card was acquired by model reference
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["language"] == "en"
    assert card_data["license"] == "bsd"
    # Validate inferred model card text
    with model_path.joinpath("model_card.md").open() as file:
        card_text = file.read()
    # verify the license file has been written
    license_file = model_path.joinpath("LICENSE.txt").read_text()
    assert len(license_file) > 0
    assert card_text.startswith("\n# I made a new model!")
    # validate MLmodel files
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "QuestionAnsweringPipeline"
    assert flavor_config["pipeline_model_type"] == "MobileBertForQuestionAnswering"
    assert flavor_config["task"] == "question-answering"
    assert flavor_config["source_model_name"] == "csarron/mobilebert-uncased-squad-v2"


def test_basic_save_model_and_load_text_pipeline(small_seq2seq_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model={
            "model": small_seq2seq_pipeline.model,
            "tokenizer": small_seq2seq_pipeline.tokenizer,
        },
        path=model_path,
    )
    loaded = mlflow.transformers.load_model(model_path)
    result = loaded("MLflow is a really neat tool!")
    assert result[0]["label"] == "happy"
    assert result[0]["score"] > 0.5


def test_basic_save_model_and_load_vision_pipeline(small_vision_model, model_path, image_for_test):
    if IS_NEW_FEATURE_EXTRACTION_API:
        model = {
            "model": small_vision_model.model,
            "image_processor": small_vision_model.image_processor,
            "tokenizer": small_vision_model.tokenizer,
        }
    else:
        model = {
            "model": small_vision_model.model,
            "feature_extractor": small_vision_model.feature_extractor,
            "tokenizer": small_vision_model.tokenizer,
        }
    mlflow.transformers.save_model(
        transformers_model=model,
        path=model_path,
    )
    loaded = mlflow.transformers.load_model(model_path)
    prediction = loaded(image_for_test)
    assert prediction[0]["label"] == "tabby, tabby cat"
    assert prediction[0]["score"] > 0.5


@flaky()
def test_multi_modal_pipeline_save_and_load(small_multi_modal_pipeline, model_path, image_for_test):
    mlflow.transformers.save_model(transformers_model=small_multi_modal_pipeline, path=model_path)
    question = "How many cats are in the picture?"
    # Load components
    components = mlflow.transformers.load_model(model_path, return_type="components")
    if IS_NEW_FEATURE_EXTRACTION_API:
        expected_components = {"model", "task", "tokenizer", "image_processor"}
    else:
        expected_components = {"model", "task", "tokenizer", "feature_extractor"}
    assert set(components.keys()).intersection(expected_components) == expected_components
    constructed_pipeline = transformers.pipeline(**components)
    answer = constructed_pipeline(image=image_for_test, question=question)
    assert answer[0]["answer"] == "2"
    # Load pipeline
    pipeline = mlflow.transformers.load_model(model_path)
    pipeline_answer = pipeline(image=image_for_test, question=question)
    assert pipeline_answer[0]["answer"] == "2"
    # Test invalid loading mode
    with pytest.raises(MlflowException, match="The specified return_type mode 'magic' is"):
        mlflow.transformers.load_model(model_path, return_type="magic")


def test_multi_modal_component_save_and_load(component_multi_modal, model_path, image_for_test):
    if IS_NEW_FEATURE_EXTRACTION_API:
        processor = component_multi_modal["image_processor"]
    else:
        processor = component_multi_modal["feature_extractor"]
    mlflow.transformers.save_model(
        transformers_model=component_multi_modal,
        path=model_path,
        processor=processor,
    )
    # Ensure that the appropriate Processor object was detected and loaded with the pipeline.
    loaded_components = mlflow.transformers.load_model(
        model_uri=model_path, return_type="components"
    )
    assert isinstance(loaded_components["model"], transformers.ViltForQuestionAnswering)
    assert isinstance(loaded_components["tokenizer"], transformers.BertTokenizerFast)
    # This is to simulate a post-processing processor that would be used externally to a Pipeline
    # This isn't being tested on an actual use case of such a model type due to the size of
    # these types of models that have this interface being ill-suited for CI testing.

    if IS_NEW_FEATURE_EXTRACTION_API:
        processor_key = "image_processor"
        assert isinstance(loaded_components[processor_key], transformers.ViltImageProcessor)
    else:
        processor_key = "feature_extractor"
        assert isinstance(loaded_components[processor_key], transformers.ViltProcessor)
        assert isinstance(loaded_components["processor"], transformers.ViltProcessor)
    if not IS_NEW_FEATURE_EXTRACTION_API:
        # NB: This simulated behavior is no longer valid in versions 4.27.4 and above.
        # With the port of functionality away from feature extractor types, the new architecture
        # for multi-modal models is entirely pipeline based.
        # Make sure that the component usage works correctly when extracted from inference loading
        model = loaded_components["model"]
        processor = loaded_components["processor"]
        question = "What are the cats doing?"
        inputs = processor(image_for_test, question, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        assert answer == "sleeping"


@flaky()
def test_pipeline_saved_model_with_processor_cannot_be_loaded_as_pipeline(
    component_multi_modal, model_path
):
    invalid_pipeline = transformers.pipeline(
        task="visual-question-answering", **component_multi_modal
    )
    if IS_NEW_FEATURE_EXTRACTION_API:
        processor = component_multi_modal["image_processor"]
    else:
        processor = component_multi_modal["feature_extractor"]
    mlflow.transformers.save_model(
        transformers_model=invalid_pipeline,
        path=model_path,
        processor=processor,  # If this is specified, we cannot guarantee correct inference
    )
    with pytest.raises(
        MlflowException, match="This model has been saved with a processor. Processor objects"
    ):
        mlflow.transformers.load_model(model_uri=model_path, return_type="pipeline")


def test_component_saved_model_with_processor_cannot_be_loaded_as_pipeline(
    component_multi_modal, model_path
):
    if IS_NEW_FEATURE_EXTRACTION_API:
        processor = component_multi_modal["image_processor"]
    else:
        processor = component_multi_modal["feature_extractor"]
    mlflow.transformers.save_model(
        transformers_model=component_multi_modal,
        path=model_path,
        processor=processor,
    )
    with pytest.raises(
        MlflowException,
        match="This model has been saved with a processor. Processor objects are not compatible "
        "with Pipelines. Please load",
    ):
        mlflow.transformers.load_model(model_uri=model_path, return_type="pipeline")


@pytest.mark.parametrize("should_start_run", [True, False])
def test_log_and_load_transformers_pipeline(small_qa_pipeline, tmp_path, should_start_run):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "transformers"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["transformers"])
        model_info = mlflow.transformers.log_model(
            transformers_model=small_qa_pipeline,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.transformers.load_model(model_uri=model_uri, return_type="pipeline")
        assert (
            reloaded_model(
                {"question": "Who's house?", "context": "The house is owned by a man named Run."}
            )["answer"]
            == "Run"
        )
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()
    finally:
        mlflow.end_run()


def test_load_pipeline_from_remote_uri_succeeds(small_seq2seq_pipeline, model_path, mock_s3_bucket):
    mlflow.transformers.save_model(transformers_model=small_seq2seq_pipeline, path=model_path)
    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)
    model_uri = os.path.join(artifact_root, artifact_path)
    loaded = mlflow.transformers.load_model(model_uri=str(model_uri), return_type="pipeline")
    assert loaded("I like it when CI checks pass and are never flaky!")[0]["label"] == "happy"


def test_transformers_log_model_calls_register_model(small_qa_pipeline, tmp_path):
    artifact_path = "transformers"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["transformers", "torch", "torchvision"])
        mlflow.transformers.log_model(
            transformers_model=small_qa_pipeline,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="Question-Answering Model 1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_uri,
            registered_model_name="Question-Answering Model 1",
        )


def test_transformers_log_model_with_no_registered_model_name(small_vision_model, tmp_path):
    if IS_NEW_FEATURE_EXTRACTION_API:
        model = {
            "model": small_vision_model.model,
            "image_processor": small_vision_model.image_processor,
            "tokenizer": small_vision_model.tokenizer,
        }
    else:
        model = {
            "model": small_vision_model.model,
            "feature_extractor": small_vision_model.feature_extractor,
            "tokenizer": small_vision_model.tokenizer,
        }

    artifact_path = "transformers"
    registered_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), registered_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "transformers"])
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_transformers_save_persists_requirements_in_mlflow_directory(
    small_qa_pipeline, model_path, transformers_custom_env
):
    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=model_path,
        conda_env=str(transformers_custom_env),
    )
    saved_pip_req_path = model_path.joinpath("requirements.txt")
    _compare_conda_env_requirements(transformers_custom_env, saved_pip_req_path)


def test_transformers_log_with_pip_requirements(small_multi_modal_pipeline, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()

    requirements_file = tmp_path.joinpath("requirements.txt")
    requirements_file.write_text("coolpackage")
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline, "model", pip_requirements=str(requirements_file)
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "coolpackage"], strict=True
        )
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            pip_requirements=[f"-r {requirements_file}", "alsocool"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "coolpackage", "alsocool"],
            strict=True,
        )
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            pip_requirements=[f"-c {requirements_file}", "constrainedcool"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "constrainedcool", "-c constraints.txt"],
            ["coolpackage"],
            strict=True,
        )


def test_transformers_log_with_extra_pip_requirements(small_multi_modal_pipeline, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_requirements = mlflow.transformers.get_default_pip_requirements(
        small_multi_modal_pipeline.model
    )
    requirements_file = tmp_path.joinpath("requirements.txt")
    requirements_file.write_text("coolpackage")
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline, "model", extra_pip_requirements=str(requirements_file)
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_requirements, "coolpackage"],
            strict=True,
        )
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            extra_pip_requirements=[f"-r {requirements_file}", "alsocool"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_requirements, "coolpackage", "alsocool"],
            strict=True,
        )
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            extra_pip_requirements=[f"-c {requirements_file}", "constrainedcool"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [
                expected_mlflow_version,
                *default_requirements,
                "constrainedcool",
                "-c constraints.txt",
            ],
            ["coolpackage"],
            strict=True,
        )


def test_transformers_log_with_duplicate_extra_pip_requirements(small_multi_modal_pipeline):
    with pytest.raises(
        MlflowException, match="The specified requirements versions are incompatible"
    ):
        with mlflow.start_run():
            mlflow.transformers.log_model(
                small_multi_modal_pipeline,
                "model",
                extra_pip_requirements=["transformers==1.1.0"],
            )


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is not None, reason="fails when accelerate is installed"
)
def test_transformers_tf_model_save_without_conda_env_uses_default_env_with_expected_dependencies(
    small_seq2seq_pipeline, model_path
):
    mlflow.transformers.save_model(small_seq2seq_pipeline, model_path)
    _assert_pip_requirements(
        model_path, mlflow.transformers.get_default_pip_requirements(small_seq2seq_pipeline.model)
    )
    pip_requirements = _get_deps_from_requirement_file(model_path)
    assert "tensorflow" in pip_requirements
    assert "torch" not in pip_requirements
    assert "accelerate" not in pip_requirements


def test_transformers_pt_model_save_without_conda_env_uses_default_env_with_expected_dependencies(
    small_qa_pipeline, model_path
):
    mlflow.transformers.save_model(small_qa_pipeline, model_path)
    _assert_pip_requirements(
        model_path, mlflow.transformers.get_default_pip_requirements(small_qa_pipeline.model)
    )
    pip_requirements = _get_deps_from_requirement_file(model_path)
    assert "tensorflow" not in pip_requirements
    assert "accelerate" in pip_requirements
    assert "torch" in pip_requirements


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is not None, reason="fails when accelerate is installed"
)
def test_transformers_pt_model_save_dependencies_without_accelerate(
    translation_pipeline, model_path
):
    mlflow.transformers.save_model(translation_pipeline, model_path)
    _assert_pip_requirements(
        model_path, mlflow.transformers.get_default_pip_requirements(translation_pipeline.model)
    )
    pip_requirements = _get_deps_from_requirement_file(model_path)
    assert "tensorflow" not in pip_requirements
    assert "accelerate" not in pip_requirements
    assert "torch" in pip_requirements


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is not None, reason="fails when accelerate is installed"
)
def test_transformers_tf_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    small_seq2seq_pipeline,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.transformers.log_model(small_seq2seq_pipeline, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(
        model_uri, mlflow.transformers.get_default_pip_requirements(small_seq2seq_pipeline.model)
    )
    pip_requirements = _get_deps_from_requirement_file(model_uri)
    assert "tensorflow" in pip_requirements
    assert "torch" not in pip_requirements
    # Accelerate installs Pytorch along with it, so it should not be present in the requirements
    assert "accelerate" not in pip_requirements


def test_transformers_pt_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    small_qa_pipeline,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.transformers.log_model(small_qa_pipeline, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(
        model_uri, mlflow.transformers.get_default_pip_requirements(small_qa_pipeline.model)
    )
    pip_requirements = _get_deps_from_requirement_file(model_uri)
    assert "tensorflow" not in pip_requirements
    assert "torch" in pip_requirements


def test_log_model_with_code_paths(small_qa_pipeline):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.transformers._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.transformers.log_model(small_qa_pipeline, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.transformers.FLAVOR_NAME)
        mlflow.transformers.load_model(model_uri)
        add_mock.assert_called()


def test_non_existent_model_card_entry(small_seq2seq_pipeline, model_path):
    with mock.patch("mlflow.transformers._fetch_model_card", return_value=None):
        mlflow.transformers.save_model(transformers_model=small_seq2seq_pipeline, path=model_path)

        contents = {item.name for item in model_path.iterdir()}
        assert not contents.intersection({"model_card.txt", "model_card_data.yaml"})


def test_huggingface_hub_not_installed(small_seq2seq_pipeline, model_path):
    with mock.patch.dict("sys.modules", {"huggingface_hub": None}):
        result = mlflow.transformers._fetch_model_card(small_seq2seq_pipeline.model.name_or_path)

        assert result is None

        mlflow.transformers.save_model(transformers_model=small_seq2seq_pipeline, path=model_path)

        contents = {item.name for item in model_path.iterdir()}
        assert not contents.intersection({"model_card.txt", "model_card_data.yaml"})

        license_data = model_path.joinpath("LICENSE.txt").read_text()
        assert license_data.rstrip().endswith("mobilebert")


def test_save_pipeline_without_defined_components(small_conversational_model, model_path):
    # This pipeline type explicitly does not have a configuration for an image_processor
    with mlflow.start_run():
        mlflow.transformers.save_model(
            transformers_model=small_conversational_model, path=model_path
        )
    pipe = mlflow.transformers.load_model(model_path)
    convo = transformers.Conversation("How are you today?")
    convo = pipe(convo)
    assert convo.generated_responses[-1] == "good"


@flaky()
def test_invalid_model_type_without_registered_name_does_not_save(model_path):
    invalid_pipeline = transformers.pipeline(task="text-generation", model="gpt2")
    del invalid_pipeline.model.name_or_path

    with pytest.raises(MlflowException, match="The submitted model type"):
        mlflow.transformers.save_model(transformers_model=invalid_pipeline, path=model_path)


def test_invalid_input_to_pyfunc_signature_output_wrapper_raises(component_multi_modal):
    with pytest.raises(MlflowException, match="The pipeline type submitted is not a valid"):
        mlflow.transformers.generate_signature_output(component_multi_modal["model"], "bogus")


@pytest.mark.parametrize(
    "inference_payload",
    [
        ({"question": "Who's house?", "context": "The house is owned by a man named Run."}),
        (
            [
                {
                    "question": "What color is it?",
                    "context": "Some people said it was green but I know that it's definitely blue",
                },
                {
                    "question": "How do the wheels go?",
                    "context": "The wheels on the bus go round and round. Round and round.",
                },
            ]
        ),
        (
            [
                {
                    "question": "What color is it?",
                    "context": "Some people said it was green but I know that it's pink.",
                },
                {
                    "context": "The people on the bus go up and down. Up and down.",
                    "question": "How do the people go?",
                },
            ]
        ),
    ],
)
def test_qa_pipeline_pyfunc_load_and_infer(small_qa_pipeline, model_path, inference_payload):
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(small_qa_pipeline, inference_payload),
    )

    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(inference_payload)

    assert isinstance(inference, list)
    assert all(isinstance(element, str) for element in inference)

    pd_input = (
        pd.DataFrame([inference_payload])
        if isinstance(inference_payload, dict)
        else pd.DataFrame(inference_payload)
    )
    pd_inference = pyfunc_loaded.predict(pd_input)

    assert isinstance(pd_inference, list)
    assert all(isinstance(element, str) for element in inference)


@pytest.mark.parametrize(
    "inference_payload",
    [
        image_url,
        str(image_file_path),
        pytest.param(
            "base64",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("4.33"),
                reason="base64 feature not present",
            ),
        ),
    ],
)
def test_vision_pipeline_pyfunc_load_and_infer(small_vision_model, model_path, inference_payload):
    if inference_payload == "base64":
        inference_payload = base64.b64encode(image_file_path.read_bytes()).decode("utf-8")
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(small_vision_model, inference_payload),
    )
    mlflow.transformers.save_model(
        transformers_model=small_vision_model,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    predictions = pyfunc_loaded.predict(inference_payload)

    transformers_loaded_model = mlflow.transformers.load_model(model_path)
    expected_predictions = transformers_loaded_model.predict(inference_payload)
    assert list(predictions.to_dict("records")[0].values()) == expected_predictions


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ("muppet keyboard type", ["A man is typing a muppet on a keyboard."]),
        (
            ["pencil draw paper", "pie apple eat"],
            # NB: The result of this test case, without inference config overrides is:
            # ["A man drawing on paper with pencil", "A man eating a pie with applies"]
            # The inference config override forces additional insertion of more grammatically
            # correct responses to validate that the inference config is being applied.
            ["A man draws a pencil on a paper.", "A man eats a pie of apples."],
        ),
    ],
)
def test_text2text_generation_pipeline_with_model_configs(
    text2text_generation_pipeline, tmp_path, data, result
):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text2text_generation_pipeline, data)
    )

    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    model_path1 = tmp_path.joinpath("model1")
    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path1,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path1)

    inference = pyfunc_loaded.predict(data)

    assert inference == result

    pd_input = pd.DataFrame([data]) if isinstance(data, str) else pd.DataFrame(data)
    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result

    model_path2 = tmp_path.joinpath("model2")
    signature_with_params = infer_signature(
        data,
        mlflow.transformers.generate_signature_output(text2text_generation_pipeline, data),
        model_config,
    )
    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path2,
        signature=signature_with_params,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path2)

    dict_inference = pyfunc_loaded.predict(
        data,
        params=model_config,
    )

    assert dict_inference == inference


def test_text2text_generation_pipeline_with_model_config_and_params(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "do_sample": True,
    }
    parameters = {"top_k": 3, "max_length": 30}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )
    signature = infer_signature(
        data,
        generated_output,
        parameters,
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # model_config and default params are all applied
    res = pyfunc_loaded.predict(data)
    applied_params = model_config.copy()
    applied_params.update(parameters)
    res2 = pyfunc_loaded.predict(data, applied_params)
    assert res == res2

    assert res != pyfunc_loaded.predict(data, {"max_length": 10})

    # Extra params are ignored
    assert res == pyfunc_loaded.predict(data, {"extra_param": "extra_value"})


def test_text2text_generation_pipeline_with_params_success(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    parameters = {"top_k": 2, "num_beams": 5, "do_sample": True}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )
    signature = infer_signature(
        data,
        generated_output,
        parameters,
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    # parameteres saved with ModelSignature is applied by default
    res = pyfunc_loaded.predict(data)
    res2 = pyfunc_loaded.predict(data, parameters)
    assert res == res2


def test_text2text_generation_pipeline_with_params_with_errors(
    text2text_generation_pipeline, model_path
):
    data = "muppet keyboard type"
    parameters = {"top_k": 2, "num_beams": 5, "invalid_param": "invalid_param", "do_sample": True}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )

    mlflow.transformers.save_model(
        text2text_generation_pipeline,
        path=model_path,
        signature=infer_signature(
            data,
            generated_output,
            parameters,
        ),
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    with pytest.raises(
        MlflowException,
        match=r"The params provided to the `predict` method are "
        r"not valid for pipeline Text2TextGenerationPipeline.",
    ):
        pyfunc_loaded.predict(data, parameters)

    # Type validation of params failure
    with pytest.raises(MlflowException, match=r"Invalid parameters found"):
        pyfunc_loaded.predict(data, {"top_k": "2"})


def test_text2text_generation_pipeline_with_inferred_schema(text2text_generation_pipeline):
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=text2text_generation_pipeline, artifact_path="my_model"
        )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_info.model_uri)

    assert pyfunc_loaded.predict("muppet board nails hammer") == [
        "A hammer with a muppet and nails on a board."
    ]


@pytest.mark.parametrize(
    "invalid_data",
    [
        ({"answer": "something", "context": ["nothing", "that", "makes", "sense"]}),
        ([{"answer": ["42"], "context": "life"}, {"unmatched": "keys", "cause": "failure"}]),
    ],
)
def test_invalid_input_to_text2text_pipeline(text2text_generation_pipeline, invalid_data):
    # Adding this validation test due to the fact that we're constructing the input to the
    # Pipeline. The Pipeline requires a format of a pseudo-dict-like string. An example of
    # a valid input string: "answer: green. context: grass is primarily green in color."
    # We generate this string from a dict or generate a list of these strings from a list of
    # dictionaries.
    with pytest.raises(MlflowException, match="An invalid type has been supplied. Please supply"):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(
                text2text_generation_pipeline, invalid_data
            ),
        )


@pytest.mark.parametrize(
    "data", ["Generative models are", (["Generative models are", "Computers are"])]
)
def test_text_generation_pipeline(text_generation_pipeline, model_path, data):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text_generation_pipeline, data)
    )

    model_config = {
        "prefix": "software",
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    mlflow.transformers.save_model(
        text_generation_pipeline,
        path=model_path,
        model_config=model_config,
        signature=signature,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)

    if isinstance(data, list):
        assert inference[0].startswith(data[0])
        assert inference[1].startswith(data[1])
    else:
        assert inference[0].startswith(data)

    pd_input = pd.DataFrame([data], index=[0]) if isinstance(data, str) else pd.DataFrame(data)
    pd_inference = pyfunc_loaded.predict(pd_input)

    if isinstance(data, list):
        assert pd_inference[0].startswith(data[0])
        assert pd_inference[1].startswith(data[1])
    else:
        assert pd_inference[0].startswith(data)


@pytest.mark.parametrize(
    "invalid_data",
    [
        ({"my_input": "something to predict"}),
        ([{"bogus_input": "invalid"}, "not_valid"]),
        (["tell me a story", {"of": "a properly configured pipeline input"}]),
    ],
)
def test_invalid_input_to_text_generation_pipeline(text_generation_pipeline, invalid_data):
    if isinstance(invalid_data, list):
        match = "If supplying a list, all values must be of string type"
    else:
        match = "The input data is of an incorrect type"
    with pytest.raises(MlflowException, match=match):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(text_generation_pipeline, invalid_data),
        )


@pytest.mark.parametrize(
    ("inference_payload", "result"),
    [
        ("Riding a <mask> on the beach is fun!", ["bike"]),
        (["If I had <mask>, I would fly to the top of a mountain"], ["wings"]),
        (
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
            ["cash", "ate"],
        ),
    ],
)
def test_fill_mask_pipeline(fill_mask_pipeline, model_path, inference_payload, result):
    signature = infer_signature(
        inference_payload,
        mlflow.transformers.generate_signature_output(fill_mask_pipeline, inference_payload),
    )

    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path, signature=signature)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(inference_payload)
    assert inference == result

    if len(inference_payload) > 1 and isinstance(inference_payload, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in inference_payload])
    elif isinstance(inference_payload, list) and len(inference_payload) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in inference_payload], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": inference_payload}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result


def test_fill_mask_pipeline_with_multiple_masks(fill_mask_pipeline, model_path):
    data = ["I <mask> the whole <mask> of <mask>", "I <mask> the whole <mask> of <mask>"]

    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)
    assert len(inference) == 2
    assert all(len(value) == 3 for value in inference)


@pytest.mark.parametrize(
    "invalid_data",
    [
        ({"a": "b"}),
        ([{"a": "b"}, [{"a": "c"}]]),
    ],
)
def test_invalid_input_to_fill_mask_pipeline(fill_mask_pipeline, invalid_data):
    if isinstance(invalid_data, list):
        match = "Invalid data submission. Ensure all"
    else:
        match = "The input data is of an incorrect type"
    with pytest.raises(MlflowException, match=match):
        infer_signature(
            invalid_data,
            mlflow.transformers.generate_signature_output(fill_mask_pipeline, invalid_data),
        )


@pytest.mark.parametrize(
    "data",
    [
        {
            "sequences": "I love the latest update to this IDE!",
            "candidate_labels": ["happy", "sad"],
        },
        {
            "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
            "candidate_labels": ["happy", "sad"],
            "hypothesis_template": "This example talks about how the dog is {}",
        },
    ],
)
def test_zero_shot_classification_pipeline(zero_shot_pipeline, model_path, data):
    # NB: The list submission for this pipeline type can accept json-encoded lists or lists within
    # the values of the dictionary.
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(zero_shot_pipeline, data)
    )

    mlflow.transformers.save_model(zero_shot_pipeline, model_path, signature=signature)

    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict(data)

    assert isinstance(inference, pd.DataFrame)
    if isinstance(data["sequences"], str):
        assert len(inference) == len(data["candidate_labels"])
    else:
        assert len(inference) == len(data["sequences"]) * len(data["candidate_labels"])


@pytest.mark.parametrize(
    ("query", "result"),
    [
        ({"query": "What should we order more of?"}, ["Apples"]),
        (
            {
                "query": [
                    "What is our highest sales?",
                    "What should we order more of?",
                ]
            },
            ["1230945.55", "Apples"],
        ),
    ],
)
def test_table_question_answering_pipeline(
    table_question_answering_pipeline, model_path, query, result
):
    table = {
        "Fruit": ["Apples", "Bananas", "Oranges", "Watermelon", "Blueberries"],
        "Sales": ["1230945.55", "86453.12", "11459.23", "8341.23", "2325.88"],
        "Inventory": ["910", "4589", "11200", "80", "3459"],
    }
    json_table = json.dumps(table)
    data = {**query, "table": json_table}
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(table_question_answering_pipeline, data)
    )

    mlflow.transformers.save_model(
        table_question_answering_pipeline, model_path, signature=signature
    )
    loaded = mlflow.pyfunc.load_model(model_path)

    inference = loaded.predict(data)
    assert inference == result

    pd_input = pd.DataFrame([data])
    pd_inference = loaded.predict(pd_input)
    assert pd_inference == result


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ("I've got a lovely bunch of coconuts!", ["Ich habe eine schöne Haufe von Kokos!"]),
        (
            [
                "I am the very model of a modern major general",
                "Once upon a time, there was a little turtle",
            ],
            [
                "Ich bin das Modell eines modernen Generals.",
                "Einmal gab es eine kleine Schildkröte.",
            ],
        ),
    ],
)
def test_translation_pipeline(translation_pipeline, model_path, data, result):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(translation_pipeline, data)
    )

    mlflow.transformers.save_model(translation_pipeline, path=model_path, signature=signature)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(data)
    assert inference == result

    if len(data) > 1 and isinstance(data, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in data])
    elif isinstance(data, list) and len(data) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in data], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": data}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    assert pd_inference == result


@pytest.mark.parametrize(
    "data",
    [
        "There once was a boy",
        ["Dolly isn't just a sheep anymore"],
        ["Baking cookies is quite easy", "Writing unittests is good for"],
    ],
)
def test_summarization_pipeline(summarizer_pipeline, model_path, data):
    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 90,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(summarizer_pipeline, data)
    )

    mlflow.transformers.save_model(
        summarizer_pipeline, path=model_path, model_config=model_config, signature=signature
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(data)
    if isinstance(data, list) and len(data) > 1:
        for i, entry in enumerate(data):
            assert inference[i].strip().startswith(entry)
    elif isinstance(data, list) and len(data) == 1:
        assert inference[0].strip().startswith(data[0])
    else:
        assert inference[0].strip().startswith(data)

    if len(data) > 1 and isinstance(data, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in data])
    elif isinstance(data, list) and len(data) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in data], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": data}, index=[0])

    pd_inference = pyfunc_loaded.predict(pd_input)
    if isinstance(data, list) and len(data) > 1:
        for i, entry in enumerate(data):
            assert pd_inference[i].strip().startswith(entry)
    elif isinstance(data, list) and len(data) == 1:
        assert pd_inference[0].strip().startswith(data[0])
    else:
        assert pd_inference[0].strip().startswith(data)


@pytest.mark.parametrize(
    "data",
    [
        "I'm telling you that Han shot first!",
        [
            "I think this sushi might have gone off",
            "That gym smells like feet, hot garbage, and sadness",
            "I love that we have a moon",
        ],
        [{"text": "test1", "text_pair": "test2"}],
        [{"text": "test1", "text_pair": "pair1"}, {"text": "test2", "text_pair": "pair2"}],
    ],
)
def test_classifier_pipeline(text_classification_pipeline, model_path, data):
    signature = infer_signature(
        data, mlflow.transformers.generate_signature_output(text_classification_pipeline, data)
    )
    mlflow.transformers.save_model(
        text_classification_pipeline, path=model_path, signature=signature
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(data)

    # verify that native transformers outputs match the pyfunc return values
    native_inference = text_classification_pipeline(data)
    inference_dict = inference.to_dict()

    if isinstance(data, str):
        assert len(inference) == 1
        assert inference_dict["label"][0] == native_inference[0]["label"]
        assert inference_dict["score"][0] == native_inference[0]["score"]
    else:
        assert len(inference) == len(data)
        for key in ["score", "label"]:
            for value in range(0, len(data)):
                assert native_inference[value][key] == inference_dict[key][value]


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            "I have a dog and his name is Willy!",
            ["PRON,VERB,DET,NOUN,CCONJ,PRON,NOUN,AUX,PROPN,PUNCT"],
        ),
        (["I like turtles"], ["PRON,VERB,NOUN"]),
        (
            ["We are the knights who say nee!", "Houston, we may have a problem."],
            [
                "PRON,AUX,DET,PROPN,PRON,VERB,INTJ,PUNCT",
                "PROPN,PUNCT,PRON,AUX,VERB,DET,NOUN,PUNCT",
            ],
        ),
    ],
)
@pytest.mark.parametrize("pipeline_name", ["ner_pipeline", "ner_pipeline_aggregation"])
def test_ner_pipeline(pipeline_name, model_path, data, result, request):
    pipeline = request.getfixturevalue(pipeline_name)

    signature = infer_signature(data, mlflow.transformers.generate_signature_output(pipeline, data))

    mlflow.transformers.save_model(pipeline, model_path, signature=signature)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict(data)

    assert inference == result

    if len(data) > 1 and isinstance(data, list):
        pd_input = pd.DataFrame([{"inputs": v} for v in data])
    elif isinstance(data, list) and len(data) == 1:
        pd_input = pd.DataFrame([{"inputs": v} for v in data], index=[0])
    else:
        pd_input = pd.DataFrame({"inputs": data}, index=[0])
    pd_inference = loaded_pyfunc.predict(pd_input)
    assert pd_inference == result


def test_conversational_pipeline(conversational_pipeline, model_path):
    signature = infer_signature(
        "Hi there!",
        mlflow.transformers.generate_signature_output(conversational_pipeline, "Hi there!"),
    )

    mlflow.transformers.save_model(conversational_pipeline, model_path, signature=signature)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)

    first_response = loaded_pyfunc.predict("What is the best way to get to Antarctica?")

    assert first_response == "The best way would be to go to space."

    second_response = loaded_pyfunc.predict("What kind of boat should I use?")

    assert second_response == "The best way to get to space would be to reach out and touch it."

    # Test that a new loaded instance has no context.
    loaded_again_pyfunc = mlflow.pyfunc.load_model(model_path)
    third_response = loaded_again_pyfunc.predict("What kind of boat should I use?")

    assert third_response == "The one with the guns."

    fourth_response = loaded_again_pyfunc.predict("Can I use it to go to the moon?")

    assert fourth_response == "Sure."


def test_qa_pipeline_pyfunc_predict(small_qa_pipeline):
    artifact_path = "qa_model"
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=small_qa_pipeline,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    inference_payload = json.dumps(
        {
            "inputs": {
                "question": [
                    "What color is it?",
                    "How do the people go?",
                    "What does the 'wolf' howl at?",
                ],
                "context": [
                    "Some people said it was green but I know that it's pink.",
                    "The people on the bus go up and down. Up and down.",
                    "The pack of 'wolves' stood on the cliff and a 'lone wolf' howled at "
                    "the moon for hours.",
                ],
            }
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [{0: "pink"}, {0: "up and down"}, {0: "the moon"}]

    inference_payload = json.dumps(
        {
            "inputs": {
                "question": "Who's house?",
                "context": "The house is owned by a man named Run.",
            }
        }
    )

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [{0: "Run"}]


@pytest.mark.parametrize(
    ("input_image", "result"),
    [
        (str(image_file_path), False),
        (image_url, False),
        ("base64", True),
        ("random string", False),
    ],
)
def test_vision_is_base64_image(input_image, result):
    if input_image == "base64":
        input_image = base64.b64encode(image_file_path.read_bytes()).decode("utf-8")
    assert _TransformersWrapper.is_base64_image(input_image) == result


@pytest.mark.parametrize(
    "inference_payload",
    [
        [str(image_file_path)],
        [image_url],
        pytest.param(
            "base64",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("4.33"),
                reason="base64 feature not present",
            ),
        ),
    ],
)
def test_vision_pipeline_pyfunc_predict(small_vision_model, inference_payload):
    if inference_payload == "base64":
        inference_payload = [
            base64.b64encode(image_file_path.read_bytes()).decode("utf-8"),
        ]
    artifact_path = "image_classification_model"

    # Log the image classification model
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=small_vision_model,
            artifact_path=artifact_path,
        )
    pyfunc_inference_payload = json.dumps({"inputs": inference_payload})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=pyfunc_inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    transformers_loaded_model = mlflow.transformers.load_model(model_info.model_uri)
    expected_predictions = transformers_loaded_model.predict(inference_payload)

    assert [list(pred.values()) for pred in predictions.to_dict("records")] == expected_predictions


def test_classifier_pipeline_pyfunc_predict(text_classification_pipeline):
    artifact_path = "text_classifier_model"
    data = [
        "I think this sushi might have gone off",
        "That gym smells like feet, hot garbage, and sadness",
        "I love that we have a moon",
        "I 'love' debugging subprocesses",
        'Quote "in" the string',
    ]
    signature = infer_signature(data)
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=text_classification_pipeline,
            artifact_path=artifact_path,
            signature=signature,
        )

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=json.dumps({"inputs": data}),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 5

    # test simple string input
    inference_payload = json.dumps({"inputs": ["testing"]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 1

    # Test the alternate TextClassificationPipeline input structure where text_pair is used
    # and ensure that model serving and direct native inference match
    inference_data = [
        {"text": "test1", "text_pair": "pair1"},
        {"text": "test2", "text_pair": "pair2"},
        {"text": "test 'quote", "text_pair": "pair 'quote'"},
    ]
    signature = infer_signature(inference_data)
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=text_classification_pipeline,
            artifact_path=artifact_path,
            signature=signature,
        )

    inference_payload = json.dumps({"inputs": inference_data})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    values_dict = values.to_dict()
    native_predict = text_classification_pipeline(inference_data)

    # validate that the pyfunc served model registers text_pair in the same manner as native
    for key in ["score", "label"]:
        for value in [0, 1]:
            assert values_dict[key][value] == native_predict[value][key]


def test_zero_shot_pipeline_pyfunc_predict(zero_shot_pipeline):
    artifact_path = "zero_shot_classifier_model"
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=zero_shot_pipeline,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    inference_payload = json.dumps(
        {
            "inputs": {
                "sequences": "My dog loves running through troughs of spaghetti "
                "with his mouth open",
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            }
        }
    )

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    # The length is 3 because it's a single row df cast to dict.
    assert len(values.to_dict()) == 3
    assert len(values.to_dict()["labels"]) == 2

    inference_payload = json.dumps(
        {
            "inputs": {
                "sequences": [
                    "My dog loves to eat spaghetti",
                    "My dog hates going to the vet",
                    "My 'hamster' loves to play with my 'friendly' dog",
                ],
                "candidate_labels": '["happy", "sad"]',
                "hypothesis_template": "This example talks about how the dog is {}",
            }
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 3
    assert len(values.to_dict()["labels"]) == 6


def test_table_question_answering_pyfunc_predict(table_question_answering_pipeline):
    artifact_path = "table_qa_model"
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=table_question_answering_pipeline,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    table = {
        "Fruit": ["Apples", "Bananas", "Oranges", "Watermelon 'small'", "Blueberries"],
        "Sales": ["1230945.55", "86453.12", "11459.23", "8341.23", "2325.88"],
        "Inventory": ["910", "4589", "11200", "80", "3459"],
    }

    inference_payload = json.dumps(
        {
            "inputs": {
                "query": "What should we order more of?",
                "table": table,
            }
        }
    )

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [{0: "Apples"}]

    inference_payload = json.dumps(
        {
            "inputs": {
                "query": [
                    "What is our highest sales?",
                    "What should we order more of?",
                    "Which 'fruit' has the 'highest' 'sales'?",
                ],
                "table": table,
            }
        }
    )
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [
        {0: "1230945.55"},
        {0: "Apples"},
        {0: "Apples"},
    ]


def test_feature_extraction_pipeline(feature_extraction_pipeline):
    sentences = ["hi", "hello"]
    signature = infer_signature(
        sentences,
        mlflow.transformers.generate_signature_output(feature_extraction_pipeline, sentences),
    )

    artifact_path = "feature_extraction_pipeline"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=feature_extraction_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=["A sentence", "Another sentence"],
        )

    # Load as native
    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)

    inference_single = "Testing"
    inference_mult = ["Testing something", "Testing something else"]

    pred = loaded_pipeline(inference_single)
    assert len(pred[0][0]) > 10
    assert isinstance(pred[0][0][0], float)

    pred_multiple = loaded_pipeline(inference_mult)
    assert len(pred_multiple[0][0]) > 2
    assert isinstance(pred_multiple[0][0][0][0], float)

    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    pyfunc_pred = loaded_pyfunc.predict(inference_single)

    assert isinstance(pyfunc_pred, np.ndarray)

    assert np.array_equal(np.array(pred[0]), pyfunc_pred)

    pyfunc_pred_multiple = loaded_pyfunc.predict(inference_mult)

    assert np.array_equal(np.array(pred_multiple[0][0]), pyfunc_pred_multiple)


def test_feature_extraction_pipeline_pyfunc_predict(feature_extraction_pipeline):
    artifact_path = "feature_extraction"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=feature_extraction_pipeline,
            artifact_path=artifact_path,
        )

    inference_payload = json.dumps({"inputs": ["sentence one", "sentence two"]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.columns) == 384
    assert len(values) == 4

    inference_payload = json.dumps({"inputs": "sentence three"})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    # A single string input is an invalid input to serving. Verify that this throws.
    with pytest.raises(MlflowException, match="Invalid response. Predictions response contents"):
        PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()


def test_loading_unsupported_pipeline_type_as_pyfunc(small_multi_modal_pipeline, model_path):
    mlflow.transformers.save_model(small_multi_modal_pipeline, model_path)
    with pytest.raises(MlflowException, match='Model does not have the "python_function" flavor'):
        mlflow.pyfunc.load_model(model_path)


def test_pyfunc_input_validations(mock_pyfunc_wrapper):
    def ensure_raises(data, match):
        with pytest.raises(MlflowException, match=match):
            mock_pyfunc_wrapper._validate_str_or_list_str(data)

    match1 = "The input data is of an incorrect type"
    match2 = "If supplying a list, all values must"
    ensure_raises({"a": "b"}, match1)
    ensure_raises(("a", "b"), match1)
    ensure_raises({"a", "b"}, match1)
    ensure_raises(True, match1)
    ensure_raises(12, match1)
    ensure_raises([1, 2, 3], match2)
    ensure_raises([{"a", "b"}], match2)
    ensure_raises([["a", "b", "c'"]], match2)
    ensure_raises([{"a": "b"}, {"a": "c"}], match2)
    ensure_raises([[1], [2]], match2)


def test_pyfunc_json_encoded_dict_parsing(mock_pyfunc_wrapper):
    plain_dict = {"a": "b", "b": "c"}
    list_dict = [plain_dict, plain_dict]

    plain_input = {"in": json.dumps(plain_dict)}
    list_input = {"in": json.dumps(list_dict)}

    plain_parsed = mock_pyfunc_wrapper._parse_json_encoded_dict_payload_to_dict(plain_input, "in")
    assert plain_parsed == {"in": plain_dict}

    list_parsed = mock_pyfunc_wrapper._parse_json_encoded_dict_payload_to_dict(list_input, "in")
    assert list_parsed == {"in": list_dict}

    invalid_parsed = mock_pyfunc_wrapper._parse_json_encoded_dict_payload_to_dict(
        plain_input, "invalid"
    )
    assert invalid_parsed != {"in": plain_dict}
    assert invalid_parsed == plain_input


def test_pyfunc_json_encoded_list_parsing(mock_pyfunc_wrapper):
    plain_list = ["a", "b", "c"]
    nested_list = [plain_list, plain_list]
    list_dict = [{"a": "b"}, {"a": "c"}]

    plain_input = {"in": json.dumps(plain_list)}
    nested_input = {"in": json.dumps(nested_list)}
    list_dict_input = {"in": json.dumps(list_dict)}

    plain_parsed = mock_pyfunc_wrapper._parse_json_encoded_list(plain_input, "in")
    assert plain_parsed == {"in": plain_list}

    nested_parsed = mock_pyfunc_wrapper._parse_json_encoded_list(nested_input, "in")
    assert nested_parsed == {"in": nested_list}

    list_dict_parsed = mock_pyfunc_wrapper._parse_json_encoded_list(list_dict_input, "in")
    assert list_dict_parsed == {"in": list_dict}

    with pytest.raises(MlflowException, match="Invalid key in inference payload. The "):
        mock_pyfunc_wrapper._parse_json_encoded_list(list_dict_input, "invalid")


def test_pyfunc_text_to_text_input(mock_pyfunc_wrapper):
    text2text_input = {"context": "a", "answer": "b"}
    parsed_input = mock_pyfunc_wrapper._parse_text2text_input(text2text_input)
    assert parsed_input == "context: a answer: b"

    text2text_input_list = [text2text_input, text2text_input]
    parsed_input_list = mock_pyfunc_wrapper._parse_text2text_input(text2text_input_list)
    assert parsed_input_list == ["context: a answer: b", "context: a answer: b"]

    parsed_with_inputs = mock_pyfunc_wrapper._parse_text2text_input({"inputs": "a"})
    assert parsed_with_inputs == ["a"]

    parsed_str = mock_pyfunc_wrapper._parse_text2text_input("a")
    assert parsed_str == "a"

    parsed_list_str = mock_pyfunc_wrapper._parse_text2text_input(["a", "b"])
    assert parsed_list_str == ["a", "b"]

    with pytest.raises(MlflowException, match="An invalid type has been supplied"):
        mock_pyfunc_wrapper._parse_text2text_input([1, 2, 3])

    with pytest.raises(MlflowException, match="An invalid type has been supplied"):
        mock_pyfunc_wrapper._parse_text2text_input([{"a": [{"b": "c"}]}])


def test_pyfunc_qa_input(mock_pyfunc_wrapper):
    single_input = {"question": "a", "context": "b"}
    parsed_single_input = mock_pyfunc_wrapper._parse_question_answer_input(single_input)
    assert parsed_single_input == single_input

    multi_input = [single_input, single_input]
    parsed_multi_input = mock_pyfunc_wrapper._parse_question_answer_input(multi_input)
    assert parsed_multi_input == multi_input

    with pytest.raises(MlflowException, match="Invalid keys were submitted. Keys must"):
        mock_pyfunc_wrapper._parse_question_answer_input({"q": "a", "c": "b"})

    with pytest.raises(MlflowException, match="An invalid type has been supplied"):
        mock_pyfunc_wrapper._parse_question_answer_input("a")

    with pytest.raises(MlflowException, match="An invalid type has been supplied"):
        mock_pyfunc_wrapper._parse_question_answer_input(["a", "b", "c"])


def test_list_of_dict_to_list_of_str_parsing(mock_pyfunc_wrapper):
    # Test with a single list of dictionaries
    output_data = [{"a": "foo"}, {"a": "bar"}, {"a": "baz"}]
    expected_output = ["foo", "bar", "baz"]
    assert (
        mock_pyfunc_wrapper._parse_lists_of_dict_to_list_of_str(output_data, "a") == expected_output
    )

    # Test with a nested list of dictionaries
    output_data = [
        {"a": "foo", "b": [{"a": "bar"}]},
        {"a": "baz", "b": [{"a": "qux"}]},
    ]
    expected_output = ["foo", "bar", "baz", "qux"]
    assert (
        mock_pyfunc_wrapper._parse_lists_of_dict_to_list_of_str(output_data, "a") == expected_output
    )

    # Test with nested list with exclusion data
    output_data = [
        {"a": "valid", "b": [{"a": "another valid"}, {"b": "invalid"}]},
        {"a": "valid 2", "b": [{"a": "another valid 2"}, {"c": "invalid"}]},
    ]
    expected_output = ["valid", "another valid", "valid 2", "another valid 2"]
    assert (
        mock_pyfunc_wrapper._parse_lists_of_dict_to_list_of_str(output_data, "a") == expected_output
    )


def test_parsing_tokenizer_output(mock_pyfunc_wrapper):
    output_data = [{"a": "b"}, {"a": "c"}, {"a": "d"}]
    expected_output = "b,c,d"
    assert mock_pyfunc_wrapper._parse_tokenizer_output(output_data, {"a"}) == expected_output

    output_data = [output_data, output_data]
    expected_output = [expected_output, expected_output]
    assert mock_pyfunc_wrapper._parse_tokenizer_output(output_data, {"a"}) == expected_output


def test_parse_list_of_multiple_dicts(mock_pyfunc_wrapper):
    output_data = [{"a": "b", "d": "f"}, {"a": "z", "d": "g"}]
    target_dict_key = "a"
    expected_output = ["b"]

    assert (
        mock_pyfunc_wrapper._parse_list_of_multiple_dicts(output_data, target_dict_key)
        == expected_output
    )

    output_data = [
        [{"a": "c", "d": "q"}, {"a": "o", "d": "q"}, {"a": "d", "d": "q"}, {"a": "e", "d": "r"}],
        [{"a": "m", "d": "s"}, {"a": "e", "d": "t"}],
    ]
    target_dict_key = "a"
    expected_output = ["c", "m"]

    assert (
        mock_pyfunc_wrapper._parse_list_of_multiple_dicts(output_data, target_dict_key)
        == expected_output
    )


def test_parse_list_output_for_multiple_candidate_pipelines(mock_pyfunc_wrapper):
    # Test with a single candidate pipeline output
    output_data = [["foo", "bar", "baz"]]
    expected_output = ["foo"]
    assert (
        mock_pyfunc_wrapper._parse_list_output_for_multiple_candidate_pipelines(output_data)
        == expected_output
    )

    # Test with multiple candidate pipeline outputs
    output_data = [
        ["foo", "bar", "baz"],
        ["qux", "quux"],
        ["corge", "grault", "garply", "waldo"],
    ]
    expected_output = ["foo", "qux", "corge"]

    assert (
        mock_pyfunc_wrapper._parse_list_output_for_multiple_candidate_pipelines(output_data)
        == expected_output
    )

    # Test with an empty list
    output_data = []
    with pytest.raises(MlflowException, match="The output of the pipeline contains no"):
        mock_pyfunc_wrapper._parse_list_output_for_multiple_candidate_pipelines(output_data)

    # Test with a nested list
    output_data = [["foo"]]
    expected_output = ["foo"]
    assert (
        mock_pyfunc_wrapper._parse_list_output_for_multiple_candidate_pipelines(output_data)
        == expected_output
    )


@pytest.mark.parametrize(
    (
        "pipeline_input",
        "pipeline_output",
        "expected_output",
        "flavor_config",
        "include_prompt",
        "collapse_whitespace",
    ),
    [
        (
            "What answers?",
            [{"generated_text": "What answers?\n\nA collection of\n\nanswers"}],
            "A collection of\n\nanswers",
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            False,
        ),
        (
            "What answers?",
            [{"generated_text": "What answers?\n\nA collection of\n\nanswers"}],
            "A collection of answers",
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            True,
        ),
        (
            "Hello!",
            [{"generated_text": "Hello!\n\nHow are you?"}],
            "How are you?",
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            False,
        ),
        (
            "Hello!",
            [{"generated_text": "Hello!\n\nA: How are you?\n\n"}],
            "How are you?",
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            True,
        ),
        (
            "Hello!",
            [{"generated_text": "Hello!\n\nA: How are you?\n\n"}],
            "Hello! A: How are you?",
            {"instance_type": "InstructionTextGenerationPipeline"},
            True,
            True,
        ),
        (
            "Hello!",
            [{"generated_text": "Hello!\n\nA: How\nare\nyou?\n\n"}],
            "How\nare\nyou?\n\n",
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            False,
        ),
        (
            ["Hi!", "What's up?"],
            [[{"generated_text": "Hi!\n\nHello there"}, {"generated_text": "Not much, and you?"}]],
            ["Hello there", "Not much, and you?"],
            {"instance_type": "InstructionTextGenerationPipeline"},
            False,
            False,
        ),
        # Tests disabling parsing of newline characters
        (
            ["Hi!", "What's up?"],
            [
                [
                    {"generated_text": "Hi!\n\nHello there"},
                    {"generated_text": "What's up?\n\nNot much, and you?"},
                ]
            ],
            ["Hi!\n\nHello there", "What's up?\n\nNot much, and you?"],
            {"instance_type": "InstructionTextGenerationPipeline"},
            True,
            False,
        ),
        (
            "Hello!",
            [{"generated_text": "Hello!\n\nHow are you?"}],
            "Hello!\n\nHow are you?",
            {"instance_type": "InstructionTextGenerationPipeline"},
            True,
            False,
        ),
        # Tests a standard TextGenerationPipeline output
        (
            ["We like to", "Open the"],
            [
                [
                    {"generated_text": "We like to party"},
                    {"generated_text": "Open the door get on the floor everybody do the dinosaur"},
                ]
            ],
            ["We like to party", "Open the door get on the floor everybody do the dinosaur"],
            {"instance_type": "TextGenerationPipeline"},
            True,
            True,
        ),
        # Tests a standard TextGenerationPipeline output with setting "include_prompt" (noop)
        (
            ["We like to", "Open the"],
            [
                [
                    {"generated_text": "We like to party"},
                    {"generated_text": "Open the door get on the floor everybody do the dinosaur"},
                ]
            ],
            ["We like to party", "Open the door get on the floor everybody do the dinosaur"],
            {"instance_type": "TextGenerationPipeline"},
            False,
            False,
        ),
        # Test TextGenerationPipeline removes whitespace
        (
            ["We like to", "Open the"],
            [
                [
                    {"generated_text": "  We like   to    party"},
                    {
                        "generated_text": "Open the   door get on the floor   everybody    "
                        "do\nthe dinosaur"
                    },
                ]
            ],
            ["We like to party", "Open the door get on the floor everybody do the dinosaur"],
            {"instance_type": "TextGenerationPipeline"},
            False,
            True,
        ),
    ],
)
def test_parse_input_from_instruction_pipeline(
    mock_pyfunc_wrapper,
    pipeline_input,
    pipeline_output,
    expected_output,
    flavor_config,
    include_prompt,
    collapse_whitespace,
):
    assert (
        mock_pyfunc_wrapper._strip_input_from_response_in_instruction_pipelines(
            pipeline_input,
            pipeline_output,
            "generated_text",
            flavor_config,
            include_prompt,
            collapse_whitespace,
        )
        == expected_output
    )


@pytest.mark.parametrize(
    "flavor_config",
    [
        {"instance_type": "InstructionTextGenerationPipeline"},
        {"instance_type": "TextGenerationPipeline"},
    ],
)
def test_invalid_instruction_pipeline_parsing(mock_pyfunc_wrapper, flavor_config):
    prompt = "What is your favorite boba flavor?"

    bad_output = {"generated_text": ["Strawberry Milk Cap", "Honeydew with boba"]}

    with pytest.raises(MlflowException, match="Unable to parse the pipeline output. Expected"):
        mock_pyfunc_wrapper._strip_input_from_response_in_instruction_pipelines(
            prompt, bad_output, "generated_text", flavor_config, True
        )


@pytest.mark.skipif(RUNNING_IN_GITHUB_ACTIONS, reason=GITHUB_ACTIONS_SKIP_REASON)
def test_instructional_pipeline_no_prompt_in_output(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # Validate removal of prompt but inclusion of newlines by default
        model_config={"max_length": 100, "include_prompt": False},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert not inference[0].startswith("What is MLflow?")
    assert "\n" in inference[0]


@pytest.mark.skipif(RUNNING_IN_GITHUB_ACTIONS, reason=GITHUB_ACTIONS_SKIP_REASON)
def test_instructional_pipeline_no_prompt_in_output_and_removal_of_newlines(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # Validate removal of prompt but inclusion of newlines by default
        model_config={"max_length": 100, "include_prompt": False, "collapse_whitespace": True},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert not inference[0].startswith("What is MLflow?")
    assert "\n" not in inference[0]


@pytest.mark.skipif(RUNNING_IN_GITHUB_ACTIONS, reason=GITHUB_ACTIONS_SKIP_REASON)
def test_instructional_pipeline_with_prompt_in_output(model_path):
    architecture = "databricks/dolly-v2-3b"
    dolly = transformers.pipeline(model=architecture, trust_remote_code=True)

    mlflow.transformers.save_model(
        transformers_model=dolly,
        path=model_path,
        # test default propagation of `include_prompt`=True and `collapse_whitespace`=False
        model_config={"max_length": 100},
        input_example="Hello, Dolly!",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict("What is MLflow?")

    assert inference[0].startswith("What is MLflow?")
    assert "\n\n" in inference[0]


@pytest.mark.skipif(not _IS_PIPELINE_DTYPE_SUPPORTED_VERSION, reason="Feature does not exist")
@flaky()
def test_load_as_pipeline_preserves_framework_and_dtype(model_path):
    task = "translation_en_to_fr"

    # Many of the 'full configuration' arguments specified are not stored as instance arguments
    # for a pipeline; rather, they are only used when acquiring the pipeline components from
    # the huggingface hub at initial pipeline creation. If a pipeline is specified, it is
    # irrelevant to store these.
    full_config_pipeline = transformers.pipeline(
        task=task,
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
        framework="pt",
        torch_dtype=torch.bfloat16,
    )

    mlflow.transformers.save_model(
        transformers_model=full_config_pipeline,
        path=model_path,
    )

    base_loaded = mlflow.transformers.load_model(model_path)
    assert base_loaded.torch_dtype == torch.bfloat16
    assert base_loaded.framework == "pt"
    assert base_loaded.model.dtype == torch.bfloat16

    loaded_pipeline = mlflow.transformers.load_model(model_path, torch_dtype=torch.float64)

    assert loaded_pipeline.torch_dtype == torch.float64
    assert loaded_pipeline.framework == "pt"
    assert loaded_pipeline.model.dtype == torch.float64

    prediction = loaded_pipeline.predict("Hello there. How are you today?")
    assert prediction[0]["translation_text"].startswith("Bonjour")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
@pytest.mark.skipif(not _IS_PIPELINE_DTYPE_SUPPORTED_VERSION, reason="Feature does not exist")
@flaky()
def test_load_pyfunc_mutate_torch_dtype(model_path, dtype):
    task = "translation_en_to_fr"

    full_config_pipeline = transformers.pipeline(
        task=task,
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
        framework="pt",
        torch_dtype=dtype,
    )

    mlflow.transformers.save_model(
        transformers_model=full_config_pipeline,
        path=model_path,
    )

    # Since we can't directly access the underlying wrapped model instance, evaluate the
    # ability to generate an inference with a specific dtype to ensure that there are no
    # complications with setting different types within pyfunc.
    loaded_pipeline = mlflow.pyfunc.load_model(model_path)

    prediction = loaded_pipeline.predict("Hello there. How are you today?")

    assert prediction[0].startswith("Bonjour")


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_save_and_load(model_path, whisper_pipeline, sound_file_for_test):
    # NB: This test validates pre-processing via converting the sounds file into the
    # appropriate bitrate encoding rate and casting to a numpy array. Other tests validate
    # the 'raw' file input of bytes.

    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }

    signature = infer_signature(
        sound_file_for_test,
        mlflow.transformers.generate_signature_output(whisper_pipeline, sound_file_for_test),
    )

    mlflow.transformers.save_model(
        transformers_model=whisper_pipeline,
        path=model_path,
        model_config=model_config,
        signature=signature,
    )

    loaded_pipeline = mlflow.transformers.load_model(model_path)

    transcription = loaded_pipeline(sound_file_for_test, **model_config)
    assert transcription["text"].startswith(" 30")

    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)

    pyfunc_transcription = json.loads(loaded_pyfunc.predict(sound_file_for_test)[0])

    assert transcription["text"] == pyfunc_transcription["text"]
    # Due to the choice of using tuples within the return type, equivalency validation for the
    # "chunks" values is not explicitly equivalent since tuples are cast to lists when json
    # serialized.
    assert transcription["chunks"][0]["text"] == pyfunc_transcription["chunks"][0]["text"]


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_signature_inference(whisper_pipeline, sound_file_for_test):
    signature = infer_signature(
        sound_file_for_test,
        mlflow.transformers.generate_signature_output(whisper_pipeline, sound_file_for_test),
    )

    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }
    complex_signature = infer_signature(
        sound_file_for_test,
        mlflow.transformers.generate_signature_output(
            whisper_pipeline, sound_file_for_test, model_config
        ),
    )

    assert signature == complex_signature


def test_whisper_model_serve_and_score_with_inferred_signature(whisper_pipeline, raw_audio_file):
    artifact_path = "whisper"

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline, artifact_path=artifact_path
        )

    # Test inputs format
    inference_payload = json.dumps({"inputs": [base64.b64encode(raw_audio_file).decode("ascii")]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.loc[0, 0].startswith("30")


def test_whisper_model_serve_and_score(whisper_pipeline, raw_audio_file):
    artifact_path = "whisper"
    signature = infer_signature(
        raw_audio_file,
        mlflow.transformers.generate_signature_output(whisper_pipeline, raw_audio_file),
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline, artifact_path=artifact_path, signature=signature
        )

    # Test inputs format
    inference_payload = json.dumps({"inputs": [base64.b64encode(raw_audio_file).decode("ascii")]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.loc[0, 0].startswith("30")

    # Test split format
    inference_df = pd.DataFrame(
        pd.Series([base64.b64encode(raw_audio_file).decode("ascii")], name="audio_file")
    )
    split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
    split_json = json.dumps(split_dict)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=split_json,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.loc[0, 0].startswith("30")

    # Test records format
    records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
    records_json = json.dumps(records_dict)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=records_json,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.loc[0, 0].startswith("30")


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_serve_and_score_with_timestamps(whisper_pipeline, raw_audio_file):
    artifact_path = "whisper_timestamps"
    signature = infer_signature(
        raw_audio_file,
        mlflow.transformers.generate_signature_output(whisper_pipeline, raw_audio_file),
    )
    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            model_config=model_config,
            input_example=raw_audio_file,
        )

    inference_payload = json.dumps({"inputs": [base64.b64encode(raw_audio_file).decode("ascii")]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    payload_output = json.loads(values.loc[0, 0])

    assert (
        payload_output["text"]
        == mlflow.transformers.load_model(model_info.model_uri)(raw_audio_file, **model_config)[
            "text"
        ]
    )


def test_audio_classification_pipeline(audio_classification_pipeline, raw_audio_file):
    artifact_path = "audio_classification"
    signature = infer_signature(
        raw_audio_file,
        mlflow.transformers.generate_signature_output(
            audio_classification_pipeline, raw_audio_file
        ),
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=audio_classification_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=raw_audio_file,
        )

    inference_payload = json.dumps({"inputs": [base64.b64encode(raw_audio_file).decode("ascii")]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    assert isinstance(values, pd.DataFrame)
    assert len(values) > 1
    assert list(values.columns) == ["score", "label"]


def test_audio_classification_with_default_schema(audio_classification_pipeline, raw_audio_file):
    artifact_path = "audio_classification"

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=audio_classification_pipeline,
            artifact_path=artifact_path,
        )

    inference_df = pd.DataFrame(
        pd.Series([base64.b64encode(raw_audio_file).decode("ascii")], name="audio")
    )
    split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
    split_json = json.dumps(split_dict)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=split_json,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    assert isinstance(values, pd.DataFrame)
    assert len(values) > 1
    assert list(values.columns) == ["score", "label"]


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_with_url(whisper_pipeline):
    artifact_path = "whisper_url"

    url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/apollo11_launch.wav"
    )

    signature = infer_signature(
        url, mlflow.transformers.generate_signature_output(whisper_pipeline, url)
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
            signature=signature,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    url_inference = pyfunc_model.predict(url)

    inference_payload = json.dumps({"inputs": [url]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    payload_output = values.loc[0, 0]

    assert url_inference[0] == payload_output


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_pyfunc_with_invalid_uri_input(whisper_pipeline):
    artifact_path = "whisper_url"

    url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/apollo11_launch.wav"
    )

    signature = infer_signature(
        url,
        mlflow.transformers.generate_signature_output(whisper_pipeline, url),
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
            signature=signature,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    bad_uri_msg = "An invalid string input was provided. String"

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("An invalid path")

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("//www.invalid.net/audio.wav")

    with pytest.raises(MlflowException, match=bad_uri_msg):
        pyfunc_model.predict("https:///my/audio.mp3")


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_using_uri_with_default_signature_raises(whisper_pipeline):
    artifact_path = "whisper_url"

    url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/apollo11_launch.wav"
    )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    url_inference = pyfunc_model.predict(url)

    assert url_inference[0].startswith("30")
    # Ensure that direct pyfunc calling even with a conflicting signature still functions
    inference_payload = json.dumps({"inputs": [url]})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    response_data = json.loads(response.content.decode("utf-8"))

    assert response_data["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Failed to process the input audio data. Either" in response_data["message"]


def test_whisper_model_with_malformed_audio(whisper_pipeline):
    artifact_path = "whisper"

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline, artifact_path=artifact_path
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    invalid_audio = b"This isn't a real audio file"

    with pytest.raises(MlflowException, match="Failed to process the input audio data. Either"):
        pyfunc_model.predict([invalid_audio])


@pytest.mark.parametrize(
    "model_name", ["tiiuae/falcon-7b", "databricks/dolly-v2-7b", "runwayml/stable-diffusion-v1-5"]
)
def test_save_model_card_with_non_utf_characters(tmp_path, model_name):
    # non-ascii unicode characters
    test_text = (
        "Emoji testing! \u2728 \U0001F600 \U0001F609 \U0001F606 "
        "\U0001F970 \U0001F60E \U0001F917 \U0001F9D0"
    )

    card_data: ModelCard = huggingface_hub.ModelCard.load(model_name)
    card_data.text = card_data.text + "\n\n" + test_text
    custom_data = card_data.data.to_dict()
    custom_data["emojis"] = test_text

    card_data.data = huggingface_hub.CardData(**custom_data)
    _write_card_data(card_data, tmp_path)

    txt = tmp_path.joinpath(_CARD_TEXT_FILE_NAME).read_text()
    assert txt == card_data.text
    data = yaml.safe_load(tmp_path.joinpath(_CARD_DATA_FILE_NAME).read_text())
    assert data == card_data.data.to_dict()


def test_vision_pipeline_pyfunc_predict_with_kwargs(small_vision_model):
    artifact_path = "image_classification_model"

    parameters = {
        "top_k": 2,
    }
    inference_payload = json.dumps(
        {
            "inputs": [image_url],
            "params": parameters,
        }
    )

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=small_vision_model,
            artifact_path=artifact_path,
            signature=infer_signature(
                image_url,
                mlflow.transformers.generate_signature_output(small_vision_model, image_url),
                params=parameters,
            ),
        )
        model_uri = model_info.model_uri
    transformers_loaded_model = mlflow.transformers.load_model(model_uri)
    expected_predictions = transformers_loaded_model.predict(image_url)

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert (
        list(predictions.to_dict("records")[0].values())
        == expected_predictions[: parameters["top_k"]]
    )


def test_qa_pipeline_pyfunc_predict_with_kwargs(small_qa_pipeline):
    artifact_path = "qa_model"
    data = {
        "question": [
            "What color is it?",
            "How do the people go?",
            "What does the 'wolf' howl at?",
        ],
        "context": [
            "Some people said it was green but I know that it's pink.",
            "The people on the bus go up and down. Up and down.",
            "The pack of 'wolves' stood on the cliff and a 'lone wolf' howled at "
            "the moon for hours.",
        ],
    }
    parameters = {
        "top_k": 2,
        "max_answer_len": 5,
    }
    inference_payload = json.dumps(
        {
            "inputs": data,
            "params": parameters,
        }
    )
    output = mlflow.transformers.generate_signature_output(small_qa_pipeline, data)
    signature_with_params = infer_signature(
        data,
        output,
        parameters,
    )
    expected_signature = ModelSignature(
        Schema(
            [
                ColSpec(Array(DataType.string), name="question"),
                ColSpec(Array(DataType.string), name="context"),
            ]
        ),
        Schema([ColSpec(DataType.string)]),
        ParamSchema(
            [
                ParamSpec("top_k", DataType.long, 2),
                ParamSpec("max_answer_len", DataType.long, 5),
            ]
        ),
    )
    assert signature_with_params == expected_signature

    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=small_qa_pipeline,
            artifact_path=artifact_path,
            signature=signature_with_params,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records") == [
        {0: "pink"},
        {0: "pink."},
        {0: "up and down"},
        {0: "Up and down"},
        {0: "the moon"},
        {0: "moon"},
    ]


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_serve_and_score_with_timestamps_with_kwargs(
    whisper_pipeline, raw_audio_file
):
    artifact_path = "whisper_timestamps"
    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }
    signature = infer_signature(
        raw_audio_file,
        mlflow.transformers.generate_signature_output(whisper_pipeline, raw_audio_file),
        params=model_config,
    )
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=raw_audio_file,
        )

    inference_payload = json.dumps(
        {
            "inputs": [base64.b64encode(raw_audio_file).decode("ascii")],
            "model_config": model_config,
        }
    )
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    payload_output = json.loads(values.loc[0, 0])

    assert (
        payload_output["text"]
        == mlflow.transformers.load_model(model_info.model_uri)(raw_audio_file, **model_config)[
            "text"
        ]
    )


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.0"), reason="Feature does not exist"
)
def test_whisper_model_serve_and_score_with_input_example_with_params(
    whisper_pipeline, raw_audio_file
):
    artifact_path = "whisper_timestamps"
    inference_config = {
        "return_timestamps": "word",
        "chunk_length_s": 20,
        "stride_length_s": [5, 3],
    }
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path=artifact_path,
            input_example=(raw_audio_file, inference_config),
        )
    # model signature inferred from input_example
    signature = infer_signature(
        raw_audio_file,
        mlflow.transformers.generate_signature_output(whisper_pipeline, raw_audio_file),
        params=inference_config,
    )
    assert model_info.signature == signature

    inference_payload = json.dumps(
        {
            "inputs": [base64.b64encode(raw_audio_file).decode("ascii")],
        }
    )
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    payload_output = json.loads(values.loc[0, 0])

    assert (
        payload_output["text"]
        == mlflow.transformers.load_model(model_info.model_uri)(raw_audio_file, **inference_config)[
            "text"
        ]
    )


def test_uri_directory_renaming_handling_pipeline(model_path, small_seq2seq_pipeline):
    with mlflow.start_run():
        mlflow.transformers.save_model(transformers_model=small_seq2seq_pipeline, path=model_path)

    absolute_model_directory = os.path.join(model_path, "model")
    renamed_to_old_convention = os.path.join(model_path, "pipeline")
    os.rename(absolute_model_directory, renamed_to_old_convention)

    # remove the 'model_binary' entries to emulate older versions of MLflow
    mlmodel_file = os.path.join(model_path, "MLmodel")
    with open(mlmodel_file) as yaml_file:
        mlmodel = yaml.safe_load(yaml_file)

    mlmodel["flavors"]["python_function"].pop("model_binary", None)
    mlmodel["flavors"]["transformers"].pop("model_binary", None)

    with open(mlmodel_file, "w") as yaml_file:
        yaml.safe_dump(mlmodel, yaml_file)

    loaded_model = mlflow.pyfunc.load_model(model_path)

    prediction = loaded_model.predict("test")
    assert isinstance(prediction, pd.DataFrame)
    assert isinstance(prediction["label"][0], str)


def test_uri_directory_renaming_handling_components(model_path, small_seq2seq_pipeline):
    components = {
        "tokenizer": small_seq2seq_pipeline.tokenizer,
        "model": small_seq2seq_pipeline.model,
    }

    with mlflow.start_run():
        mlflow.transformers.save_model(transformers_model=components, path=model_path)

    absolute_model_directory = os.path.join(model_path, "model")
    renamed_to_old_convention = os.path.join(model_path, "pipeline")
    os.rename(absolute_model_directory, renamed_to_old_convention)

    # remove the 'model_binary' entries to emulate older versions of MLflow
    mlmodel_file = os.path.join(model_path, "MLmodel")
    with open(mlmodel_file) as yaml_file:
        mlmodel = yaml.safe_load(yaml_file)

    mlmodel["flavors"]["python_function"].pop("model_binary", None)
    mlmodel["flavors"]["transformers"].pop("model_binary", None)

    with open(mlmodel_file, "w") as yaml_file:
        yaml.safe_dump(mlmodel, yaml_file)

    loaded_model = mlflow.pyfunc.load_model(model_path)

    prediction = loaded_model.predict("test")
    assert isinstance(prediction, pd.DataFrame)
    assert isinstance(prediction["label"][0], str)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.29.2"), reason="Feature does not exist"
)
def test_whisper_model_supports_timestamps(raw_audio_file, whisper_pipeline):
    model_config = {
        "return_timestamps": "word",
        "chunk_length_s": 60,
        "batch_size": 1,
    }

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=whisper_pipeline,
            artifact_path="model",
            model_config=model_config,
        )

    model_uri = model_info.model_uri
    whisper = mlflow.transformers.load_model(model_uri)

    prediction = whisper(raw_audio_file, **model_config)
    whisper_pyfunc = mlflow.pyfunc.load_model(model_uri)
    prediction_inference = json.loads(whisper_pyfunc.predict(raw_audio_file)[0])

    first_timestamp = prediction["chunks"][0]["timestamp"]
    assert isinstance(first_timestamp, tuple)
    assert prediction_inference["chunks"][0]["timestamp"][1] == first_timestamp[1]


def test_pyfunc_model_log_load_with_artifacts_snapshot():
    architecture = "prajjwal1/bert-tiny"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.BertForQuestionAnswering.from_pretrained(architecture)
    bert_tiny_pipeline = transformers.pipeline(
        task="question-answering", model=model, tokenizer=tokenizer
    )

    class QAModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            """
            This method initializes the tokenizer and language model
            using the specified snapshot location.
            """
            snapshot_location = context.artifacts["bert-tiny-model"]
            # Initialize tokenizer and language model
            tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot_location)
            model = transformers.BertForQuestionAnswering.from_pretrained(snapshot_location)
            self.pipeline = transformers.pipeline(
                task="question-answering", model=model, tokenizer=tokenizer
            )

        def predict(self, context, model_input, params=None):
            question = model_input["question"][0]
            if isinstance(question, np.ndarray):
                question = question.item()
            ctx = model_input["context"][0]
            if isinstance(ctx, np.ndarray):
                ctx = ctx.item()
            return self.pipeline(question=question, context=ctx)

    data = {"question": "Who's house?", "context": "The house is owned by Run."}
    pyfunc_artifact_path = "question_answering_model"
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path=pyfunc_artifact_path,
            python_model=QAModel(),
            artifacts={"bert-tiny-model": "hf:/prajjwal1/bert-tiny"},
            input_example=data,
            signature=infer_signature(
                data, mlflow.transformers.generate_signature_output(bert_tiny_pipeline, data)
            ),
            extra_pip_requirements=["transformers", "torch", "numpy"],
        )

        pyfunc_model_uri = f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
        assert model_info.model_uri == pyfunc_model_uri
        pyfunc_model_path = _download_artifact_from_uri(
            f"runs:/{run.info.run_id}/{pyfunc_artifact_path}"
        )
        assert len(os.listdir(os.path.join(pyfunc_model_path, "artifacts"))) != 0
        model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=pyfunc_model_uri)
    assert model_config.to_yaml() == loaded_pyfunc_model.metadata.to_yaml()
    assert loaded_pyfunc_model.predict(data)["answer"] != ""

    # Test model serving
    inference_payload = json.dumps({"inputs": data})
    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert values.to_dict(orient="records")[0]["answer"] != ""


def test_pyfunc_model_log_load_with_artifacts_snapshot_errors():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run():
        with pytest.raises(
            MlflowException,
            match=r"Failed to download snapshot from Hugging Face Hub "
            r"with artifact_uri: hf:/invalid-repo-id.",
        ):
            mlflow.pyfunc.log_model(
                artifact_path="pyfunc_artifact_path",
                python_model=TestModel(),
                artifacts={"some-model": "hf:/invalid-repo-id"},
            )


def test_model_distributed_across_devices():
    mock_model = mock.Mock()
    mock_model.device.type = "meta"
    mock_model.hf_device_map = {
        "layer1": mock.Mock(type="cpu"),
        "layer2": mock.Mock(type="cpu"),
        "layer3": mock.Mock(type="gpu"),
        "layer4": mock.Mock(type="disk"),
    }

    assert _is_model_distributed_in_memory(mock_model)


def test_model_on_single_device():
    mock_model = mock.Mock()
    mock_model.device.type = "cpu"
    mock_model.hf_device_map = {}

    assert not _is_model_distributed_in_memory(mock_model)


def test_basic_model_with_accelerate_device_mapping_fails_save(tmp_path, model_path):
    task = "translation_en_to_de"
    architecture = "t5-small"
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=architecture,
        device_map={"shared": "cpu", "encoder": "cpu", "decoder": "disk", "lm_head": "disk"},
        offload_folder=str(tmp_path / "weights"),
        low_cpu_mem_usage=True,
    )

    tokenizer = transformers.T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=architecture, model_max_length=100
    )
    pipeline = transformers.pipeline(task=task, model=model, tokenizer=tokenizer)

    with pytest.raises(
        MlflowException,
        match="The model that is attempting to be saved has been loaded into memory",
    ):
        mlflow.transformers.save_model(transformers_model=pipeline, path=model_path)


def test_basic_model_with_accelerate_homogeneous_mapping_works(model_path):
    task = "translation_en_to_de"
    architecture = "t5-small"
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=architecture,
        device_map={"shared": "cpu", "encoder": "cpu", "decoder": "cpu", "lm_head": "cpu"},
        low_cpu_mem_usage=True,
    )

    tokenizer = transformers.T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=architecture, model_max_length=100
    )
    pipeline = transformers.pipeline(task=task, model=model, tokenizer=tokenizer)

    mlflow.transformers.save_model(transformers_model=pipeline, path=model_path)

    loaded = mlflow.transformers.load_model(model_path)
    text = "Apples are delicious"
    assert loaded(text) == pipeline(text)


def test_qa_model_model_size_bytes(small_qa_pipeline, tmp_path):
    def _calculate_expected_size(path_or_dir):
        # this helper function does not consider subdirectories
        expected_size = 0
        if path_or_dir.is_dir():
            for path in path_or_dir.iterdir():
                if not path.is_file():
                    continue
                expected_size += path.stat().st_size
        elif path_or_dir.is_file():
            expected_size = path_or_dir.stat().st_size
        return expected_size

    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=tmp_path,
    )

    # expected size only counts for files saved before the MLmodel file is saved
    model_dir = tmp_path.joinpath("model")
    tokenizer_dir = tmp_path.joinpath("components").joinpath("tokenizer")
    expected_size = 0
    for folder in [model_dir, tokenizer_dir]:
        expected_size += _calculate_expected_size(folder)
    other_files = ["model_card.md", "model_card_data.yaml", "LICENSE.txt"]
    for file in other_files:
        path = tmp_path.joinpath(file)
        expected_size += _calculate_expected_size(path)

    mlmodel = yaml.safe_load(tmp_path.joinpath("MLmodel").read_bytes())
    assert mlmodel["model_size_bytes"] == expected_size


@pytest.mark.parametrize("task", ["llm/v1/completions", "llm/v1/chat"])
def test_text_generation_save_model_with_inference_task(task, text_generation_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task=task,
    )

    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())

    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["inference_task"] == task

    assert mlmodel["metadata"]["task"] == task


def test_text_generation_save_model_with_invalid_inference_task(
    text_generation_pipeline, model_path
):
    with pytest.raises(
        MlflowException, match=r"The task provided is invalid.*Must be.*llm/v1/completions"
    ):
        mlflow.transformers.save_model(
            transformers_model=text_generation_pipeline,
            path=model_path,
            task="llm/v1/invalid",
        )


def test_text_generation_task_completions_predict_with_max_tokens(
    text_generation_pipeline, model_path
):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/completions",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?", "max_tokens": 10},
    )

    assert isinstance(inference[0], dict)
    assert inference[0]["model"] == "distilgpt2"
    assert inference[0]["object"] == "text_completion"
    assert (
        inference[0]["choices"][0]["finish_reason"] == "length"
        and inference[0]["usage"]["completion_tokens"] == 10
    ) or (
        inference[0]["choices"][0]["finish_reason"] == "stop"
        and inference[0]["usage"]["completion_tokens"] < 10
    )


def test_text_generation_task_completions_predict_with_stop(text_generation_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/completions",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(
        {"prompt": "How to learn Python in 3 weeks?", "stop": ["Python"]},
    )

    assert inference[0]["choices"][0]["finish_reason"] == "stop"
    assert (
        inference[0]["choices"][0]["text"].endswith("Python")
        or "Python" not in inference[0]["choices"][0]["text"]
    )


def test_text_generation_task_completions_serve(text_generation_pipeline):
    data = {"prompt": "How to learn Python in 3 weeks?"}

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=text_generation_pipeline,
            artifact_path="model",
            task="llm/v1/completions",
        )

    inference_payload = json.dumps({"inputs": data})

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    output_dict = values.to_dict("records")[0]
    assert output_dict["choices"][0]["text"] is not None
    assert output_dict["choices"][0]["finish_reason"] == "stop"
    assert output_dict["usage"]["prompt_tokens"] < 20


def test_model_config_is_not_mutated_after_prediction(text2text_generation_pipeline):
    # max_length and max_new_tokens cannot be used together in Transformers earlier than 4.27
    validate_max_new_tokens = Version(transformers.__version__) > Version("4.26.1")

    model_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
    }
    if validate_max_new_tokens:
        model_config["max_new_tokens"] = 500

    # Params will be used to override the values of model_config but should not mutate it
    params = {
        "top_k": 30,
        "max_length": 500,
    }
    if validate_max_new_tokens:
        params["max_new_tokens"] = 5

    pyfunc_model = _TransformersWrapper(text2text_generation_pipeline, model_config=model_config)
    assert pyfunc_model.model_config["top_k"] == 2

    prediction_output = pyfunc_model.predict(
        "rocket moon ship astronaut space gravity", params=params
    )

    assert pyfunc_model.model_config["top_k"] == 2
    assert pyfunc_model.model_config["num_beams"] == 5
    assert pyfunc_model.model_config["max_length"] == 30
    if validate_max_new_tokens:
        assert pyfunc_model.model_config["max_new_tokens"] == 500
        assert len(prediction_output[0].split(" ")) <= 5


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.34.0"), reason="Feature does not exist"
)
def test_text_generation_task_chat_predict(text_generation_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=text_generation_pipeline,
        path=model_path,
        task="llm/v1/chat",
    )

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(
        {
            "messages": [
                {"role": "system", "content": "Hello, how can I help you today?"},
                {"role": "user", "content": "How to learn Python in 3 weeks?"},
            ],
            "max_tokens": 10,
        }
    )

    assert inference[0]["choices"][0]["message"]["role"] == "assistant"
    assert (
        inference[0]["choices"][0]["finish_reason"] == "length"
        and inference[0]["usage"]["completion_tokens"] == 10
    ) or (
        inference[0]["choices"][0]["finish_reason"] == "stop"
        and inference[0]["usage"]["completion_tokens"] < 10
    )


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.34.0"), reason="Feature does not exist"
)
def test_text_generation_task_chat_serve(text_generation_pipeline):
    data = {
        "messages": [
            {"role": "user", "content": "How to learn Python in 3 weeks?"},
        ],
        "max_tokens": 10,
    }

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=text_generation_pipeline,
            artifact_path="model",
            task="llm/v1/chat",
        )

    inference_payload = json.dumps(data)

    response = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    output_dict = json.loads(response.content)[0]
    assert output_dict["choices"][0]["message"] is not None
    assert (
        output_dict["choices"][0]["finish_reason"] == "length"
        and output_dict["usage"]["completion_tokens"] == 10
    ) or (
        output_dict["choices"][0]["finish_reason"] == "stop"
        and output_dict["usage"]["completion_tokens"] < 10
    )
    assert output_dict["usage"]["prompt_tokens"] < 20


HF_COMMIT_HASH_PATTERN = re.compile(r"^[a-z0-9]{40}$")


@pytest.mark.parametrize(
    ("model_fixture", "input_example", "components"),
    [
        ("text2text_generation_pipeline", "What is MLflow?", {"tokenizer"}),
        ("text_generation_pipeline", "What is MLflow?", {"tokenizer"}),
        (
            "small_vision_model",
            image_url,
            {"feature_extractor", "image_processor"}
            if IS_NEW_FEATURE_EXTRACTION_API
            else {"feature_extractor"},
        ),
        (
            "component_multi_modal",
            {"text": "What is MLflow?", "image": image_url},
            {"image_processor", "tokenizer"}
            if IS_NEW_FEATURE_EXTRACTION_API
            else {"feature_extractor", "tokenizer"},
        ),
        ("fill_mask_pipeline", "The quick brown <mask> jumps over the lazy dog.", {"tokenizer"}),
        ("whisper_pipeline", read_raw_audio_file, {"feature_extractor", "tokenizer"}),
        ("feature_extraction_pipeline", "What is MLflow?", {"tokenizer"}),
    ],
)
def test_save_and_load_pipeline_without_save_pretrained_false(
    model_fixture, input_example, components, model_path, request
):
    pipeline = request.getfixturevalue(model_fixture)
    model = pipeline["model"] if isinstance(pipeline, dict) else pipeline.model

    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=model_path,
        save_pretrained=False,
    )

    # No weights should be saved
    assert not model_path.joinpath("model").exists()
    assert not model_path.joinpath("components").exists()

    # Validate the contents of MLModel file
    mlmodel = Model.load(str(model_path.joinpath("MLmodel")))
    flavor_conf = mlmodel.flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert flavor_conf["source_model_name"] == model.name_or_path
    assert HF_COMMIT_HASH_PATTERN.match(flavor_conf["source_model_revision"])
    assert set(flavor_conf["components"]) == components
    for c in components:
        component = pipeline[c] if isinstance(pipeline, dict) else getattr(pipeline, c)
        assert flavor_conf[f"{c}_name"] == getattr(component, "name_or_path", model.name_or_path)
        assert HF_COMMIT_HASH_PATTERN.match(flavor_conf[f"{c}_revision"])

    # Validate pyfunc load and prediction (if pyfunc supported)
    if "python_function" in mlmodel.flavors:
        if callable(input_example):
            input_example = input_example()
        mlflow.pyfunc.load_model(model_path).predict(input_example)


# Patch tempdir just to verify the invocation
@mock.patch("mlflow.transformers.TempDir", side_effect=mlflow.utils.file_utils.TempDir)
def test_persist_pretrained_model(mock_tmpdir, small_seq2seq_pipeline):
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=small_seq2seq_pipeline,
            artifact_path="model",
            save_pretrained=False,
            pip_requirements=["mlflow"],  # For speed up logging
        )

    artifact_path = Path(mlflow.artifacts.download_artifacts(model_info.model_uri))
    model_path = artifact_path / "model"
    tokenizer_path = artifact_path / "components" / "tokenizer"

    original_config = Model.load(artifact_path).flavors["transformers"]
    assert "model_binary" not in original_config
    assert "source_model_revision" in original_config
    assert not model_path.exists()
    assert not tokenizer_path.exists()

    mlflow.transformers.persist_pretrained_model(model_info.model_uri)

    mock_tmpdir.assert_called_once()
    updated_config = Model.load(model_info.model_uri).flavors["transformers"]
    assert "model_binary" in updated_config
    assert "source_model_revision" not in updated_config
    assert model_path.exists()
    assert (model_path / "tf_model.h5").exists()
    assert tokenizer_path.exists()
    assert (tokenizer_path / "tokenizer.json").exists()

    # Repeat persisting the model will no-op
    mock_tmpdir.reset_mock()
    mlflow.transformers.persist_pretrained_model(model_info.model_uri)
    mock_tmpdir.assert_not_called()

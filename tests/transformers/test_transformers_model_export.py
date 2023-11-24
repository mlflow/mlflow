import base64
import gc
import importlib.util
import json
import logging
import os
import pathlib
import textwrap
import time
from functools import wraps
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
from huggingface_hub import ModelCard, scan_cache_dir
from packaging.version import Version
from PIL import Image

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.deployments import PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.transformers import (
    _CARD_DATA_FILE_NAME,
    _CARD_TEXT_FILE_NAME,
    _FRAMEWORK_KEY,
    _INSTANCE_TYPE_KEY,
    _MODEL_PATH_OR_NAME_KEY,
    _PIPELINE_MODEL_TYPE_KEY,
    _TASK_KEY,
    _build_pipeline_from_model_input,
    _fetch_model_card,
    _generate_base_flavor_configuration,
    _get_base_model_architecture,
    _get_instance_type,
    _get_or_infer_task_type,
    _infer_transformers_task_type,
    _is_model_distributed_in_memory,
    _record_pipeline_components,
    _should_add_pyfunc_to_model,
    _TransformersModel,
    _validate_transformers_task_type,
    _write_card_data,
    get_default_conda_env,
    get_default_pip_requirements,
)
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

pytestmark = pytest.mark.large

transformers_version = Version(transformers.__version__)
_FEATURE_EXTRACTION_API_CHANGE_VERSION = "4.27.0"
_IMAGE_PROCESSOR_API_CHANGE_VERSION = "4.26.0"

# NB: Some pipelines under test in this suite come very close or outright exceed the
# default runner containers specs of 7GB RAM. Due to this inability to run the suite without
# generating a SIGTERM Error (143), some tests are marked as local only.
# See: https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted- \
# runners#supported-runners-and-hardware-resources for instance specs.
RUNNING_IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
GITHUB_ACTIONS_SKIP_REASON = "Test consumes too much memory"
image_url = "https://github.com/mlflow/mlflow/blob/master/tests/datasets/cat_image.jpg"
# Test that can only be run locally:
# - Summarization pipeline tests
# - TextClassifier pipeline tests
# - Text2TextGeneration pipeline tests
# - Conversational pipeline tests

_logger = logging.getLogger(__name__)


def flaky(max_tries=3):
    """
    Annotation decorator for retrying flaky functions up to max_tries times, and raise the Exception
    if it fails after max_tries attempts.
    :param max_tries: Maximum number of times to retry the function.
    :return: Decorated function.
    """

    def flaky_test_func(test_func):
        @wraps(test_func)
        def decorated_func(*args, **kwargs):
            for i in range(max_tries):
                try:
                    return test_func(*args, **kwargs)
                except Exception as e:
                    _logger.warning(f"Attempt {i+1} failed with error: {e}")
                    if i == max_tries - 1:
                        raise
                    time.sleep(3)

        return decorated_func

    return flaky_test_func


@pytest.fixture(autouse=True)
def force_gc():
    # This reduces the memory pressure for the usage of the larger pipeline fixtures ~500MB - 1GB
    gc.disable()
    gc.collect()
    gc.set_threshold(0)
    gc.collect()
    gc.enable()


@pytest.fixture(autouse=True)
def clean_cache(request):
    # This function will clean the cache that HuggingFace uses to limit the number of fetches from
    # the hub repository when instantiating components (tokenizers, models, etc.). Due to the
    # runner limitations for CI (As of April 2023, the runner image ubuntu-22.04 has a maximum of
    # 14GB of storage space on the provided SSDs and 7GB of RAM which are both insufficient to run
    # all validations of this test suite due to the model sizes.
    # This fixture will clear the cache iff the cache storage is > 2GB when called.
    if "skipcacheclean" in request.keywords:
        return
    else:
        full_cache = scan_cache_dir()
        cache_size_in_gb = full_cache.size_on_disk / 1000**3

        if cache_size_in_gb > 2:
            commits_to_purge = [
                rev.commit_hash for repo in full_cache.repos for rev in repo.revisions
            ]
            delete_strategy = full_cache.delete_revisions(*commits_to_purge)
            delete_strategy.execute()


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


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
def small_seq2seq_pipeline():
    # The return type of this model's language head is a List[Dict[str, Any]]
    architecture = "lordtt13/emo-mobilebert"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.TFAutoModelForSequenceClassification.from_pretrained(architecture)
    return transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@pytest.fixture
@flaky()
def small_qa_pipeline():
    # The return type of this model's language head is a Dict[str, Any]
    architecture = "csarron/mobilebert-uncased-squad-v2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture, low_cpu_mem_usage=True)
    model = transformers.MobileBertForQuestionAnswering.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    return transformers.pipeline(task="question-answering", model=model, tokenizer=tokenizer)


@pytest.fixture
@flaky()
def small_vision_model():
    architecture = "google/mobilenet_v2_1.0_224"
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    model = transformers.MobileNetV2ForImageClassification.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    return transformers.pipeline(
        task="image-classification", model=model, feature_extractor=feature_extractor
    )


@pytest.fixture
@flaky()
def small_multi_modal_pipeline():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    return transformers.pipeline(model=architecture)


@pytest.fixture
@flaky()
def component_multi_modal():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    tokenizer = transformers.BertTokenizerFast.from_pretrained(architecture, low_cpu_mem_usage=True)
    processor = transformers.ViltProcessor.from_pretrained(architecture, low_cpu_mem_usage=True)
    image_processor = transformers.ViltImageProcessor.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    model = transformers.ViltForQuestionAnswering.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    transformers_model = {"model": model, "tokenizer": tokenizer}
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        transformers_model["image_processor"] = image_processor
    else:
        transformers_model["feature_extractor"] = processor
    return transformers_model


@pytest.fixture
@flaky()
def small_conversational_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small", low_cpu_mem_usage=True
    )
    model = transformers.AutoModelWithLMHead.from_pretrained(
        "satvikag/chatbot", low_cpu_mem_usage=True
    )
    return transformers.pipeline(task="conversational", model=model, tokenizer=tokenizer)


@pytest.fixture
@flaky()
def fill_mask_pipeline():
    architecture = "distilroberta-base"
    model = transformers.AutoModelForMaskedLM.from_pretrained(architecture, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task="fill-mask", model=model, tokenizer=tokenizer)


@pytest.fixture
@flaky()
def text2text_generation_pipeline():
    task = "text2text-generation"
    architecture = "mrm8488/t5-small-finetuned-common_gen"
    model = transformers.T5ForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.T5TokenizerFast.from_pretrained(architecture)

    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@flaky()
def text_generation_pipeline():
    task = "text-generation"
    architecture = "distilgpt2"
    model = transformers.AutoModelWithLMHead.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

    return transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
    )


@pytest.fixture
@flaky()
def translation_pipeline():
    return transformers.pipeline(
        task="translation_en_to_de",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
    )


@pytest.fixture
@flaky()
def summarizer_pipeline():
    task = "summarization"
    architecture = "sshleifer/distilbart-cnn-6-6"
    model = transformers.BartForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@flaky()
def text_classification_pipeline():
    task = "text-classification"
    architecture = "distilbert-base-uncased-finetuned-sst-2-english"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@flaky()
def zero_shot_pipeline():
    task = "zero-shot-classification"
    architecture = "typeform/distilbert-base-uncased-mnli"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@flaky()
def table_question_answering_pipeline():
    return transformers.pipeline(
        task="table-question-answering", model="google/tapas-tiny-finetuned-wtq"
    )


@pytest.fixture
@flaky()
def ner_pipeline():
    return transformers.pipeline(
        task="token-classification", model="vblagoje/bert-english-uncased-finetuned-pos"
    )


@pytest.fixture
@flaky()
def ner_pipeline_aggregation():
    # Modification to the default aggregation_strategy of `None` changes the output keys in each
    # of the dictionaries. This fixture allows for testing that the correct data is extracted
    # as a return value
    return transformers.pipeline(
        task="token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="average",
    )


@pytest.fixture
@flaky()
def conversational_pipeline():
    return transformers.pipeline(model="AVeryRealHuman/DialoGPT-small-TonyStark")


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
    datasets_path = pathlib.Path(__file__).resolve().parent.parent.joinpath("datasets")

    return datasets_path.joinpath("apollo11_launch.wav").read_bytes()


@pytest.fixture
@flaky()
def whisper_pipeline():
    task = "automatic-speech-recognition"
    architecture = "openai/whisper-tiny"
    model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
    if Version(transformers.__version__) > Version("4.30.2"):
        model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
    return transformers.pipeline(
        task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
    )


@pytest.fixture
@flaky()
def audio_classification_pipeline():
    return transformers.pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")


@pytest.fixture
@flaky()
def feature_extraction_pipeline():
    st_arch = "sentence-transformers/all-MiniLM-L6-v2"
    model = transformers.AutoModel.from_pretrained(st_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(st_arch)

    return transformers.pipeline(model=model, tokenizer=tokenizer, task="feature-extraction")


def test_dependencies_pytorch(small_qa_pipeline):
    pip_requirements = get_default_pip_requirements(small_qa_pipeline.model)
    expected_requirments = {"transformers", "torch", "torchvision"}
    assert {package.split("=")[0] for package in pip_requirements}.intersection(
        expected_requirments
    ) == expected_requirments
    conda_requirements = get_default_conda_env(small_qa_pipeline.model)
    pip_in_conda = {
        package.split("=")[0] for package in conda_requirements["dependencies"][2]["pip"]
    }
    expected_conda = {"mlflow"}
    expected_conda.update(expected_requirments)
    assert pip_in_conda.intersection(expected_conda) == expected_conda


def test_dependencies_tensorflow(small_seq2seq_pipeline):
    pip_requirements = get_default_pip_requirements(small_seq2seq_pipeline.model)
    expected_requirments = {"transformers", "tensorflow"}
    assert {package.split("=")[0] for package in pip_requirements}.intersection(
        expected_requirments
    ) == expected_requirments
    conda_requirements = get_default_conda_env(small_seq2seq_pipeline.model)
    pip_in_conda = {
        package.split("=")[0] for package in conda_requirements["dependencies"][2]["pip"]
    }
    expected_conda = {"mlflow"}
    expected_conda.update(expected_requirments)
    assert pip_in_conda.intersection(expected_conda) == expected_conda


def test_task_inference(small_seq2seq_pipeline):
    expected_task = "text-classification"
    assert _infer_transformers_task_type(small_seq2seq_pipeline) == expected_task

    assert (
        _infer_transformers_task_type(
            _TransformersModel.from_dict(**{"model": small_seq2seq_pipeline.model})
        )
        == expected_task
    )
    with pytest.raises(MlflowException, match="The provided model type"):
        _infer_transformers_task_type(small_seq2seq_pipeline.tokenizer)


def test_task_validation():
    with pytest.raises(MlflowException, match="The task provided is invalid. 'fake-task' is not"):
        _validate_transformers_task_type("fake-task")
    _validate_transformers_task_type("image-classification")


def test_instance_extraction(small_qa_pipeline):
    assert _get_instance_type(small_qa_pipeline) == "QuestionAnsweringPipeline"
    assert _get_instance_type(small_qa_pipeline.model) == "MobileBertForQuestionAnswering"


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
    components = _TransformersModel.from_dict(**component_multi_modal)
    task = _infer_transformers_task_type(components)

    pipeline = _build_pipeline_from_model_input(components, task=task)
    assert not _should_add_pyfunc_to_model(pipeline)


def test_model_architecture_extraction(small_seq2seq_pipeline):
    assert _get_base_model_architecture(small_seq2seq_pipeline) == "lordtt13/emo-mobilebert"
    assert (
        _get_base_model_architecture({"model": small_seq2seq_pipeline.model})
        == "lordtt13/emo-mobilebert"
    )


def test_base_flavor_configuration_generation(small_seq2seq_pipeline, small_qa_pipeline):
    expected_seq_pipeline_conf = {
        _TASK_KEY: "text-classification",
        _INSTANCE_TYPE_KEY: "TextClassificationPipeline",
        _PIPELINE_MODEL_TYPE_KEY: "TFMobileBertForSequenceClassification",
        _MODEL_PATH_OR_NAME_KEY: "lordtt13/emo-mobilebert",
        _FRAMEWORK_KEY: "tf",
    }
    expected_qa_pipeline_conf = {
        _TASK_KEY: "question-answering",
        _INSTANCE_TYPE_KEY: "QuestionAnsweringPipeline",
        _PIPELINE_MODEL_TYPE_KEY: "MobileBertForQuestionAnswering",
        _MODEL_PATH_OR_NAME_KEY: "csarron/mobilebert-uncased-squad-v2",
        _FRAMEWORK_KEY: "pt",
    }
    seq_conf_infer_task = _generate_base_flavor_configuration(
        small_seq2seq_pipeline, _get_or_infer_task_type(small_seq2seq_pipeline)
    )
    assert seq_conf_infer_task == expected_seq_pipeline_conf
    seq_conf_specify_task = _generate_base_flavor_configuration(
        small_seq2seq_pipeline, "text-classification"
    )
    assert seq_conf_specify_task == expected_seq_pipeline_conf
    qa_conf_infer_task = _generate_base_flavor_configuration(
        small_qa_pipeline, _get_or_infer_task_type(small_qa_pipeline)
    )
    assert qa_conf_infer_task == expected_qa_pipeline_conf
    qa_conf_specify_task = _generate_base_flavor_configuration(
        small_qa_pipeline, "question-answering"
    )
    assert qa_conf_specify_task == expected_qa_pipeline_conf
    with pytest.raises(MlflowException, match="The task provided is invalid. 'magic' is not"):
        _generate_base_flavor_configuration(small_qa_pipeline, "magic")


def test_pipeline_construction_from_base_nlp_model(small_qa_pipeline):
    generated = _build_pipeline_from_model_input(
        _TransformersModel.from_dict(
            **{"model": small_qa_pipeline.model, "tokenizer": small_qa_pipeline.tokenizer}
        ),
        "question-answering",
    )
    assert isinstance(generated, type(small_qa_pipeline))
    assert isinstance(generated.tokenizer, type(small_qa_pipeline.tokenizer))


def test_pipeline_construction_from_base_vision_model(small_vision_model):
    model = {"model": small_vision_model.model, "tokenizer": small_vision_model.tokenizer}
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        model.update({"image_processor": small_vision_model.feature_extractor})
    else:
        model.update({"feature_extractor": small_vision_model.feature_extractor})
    generated = _build_pipeline_from_model_input(
        _TransformersModel.from_dict(**model),
        "image-classification",
    )
    assert isinstance(generated, type(small_vision_model))
    assert isinstance(generated.tokenizer, type(small_vision_model.tokenizer))
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        compare_type = generated.image_processor
    else:
        compare_type = generated.feature_extractor
    assert isinstance(compare_type, transformers.MobileNetV2ImageProcessor)


def test_pipeline_construction_fails_with_invalid_type(small_vision_model):
    with pytest.raises(
        MlflowException,
        match="The model type submitted is not compatible with the transformers flavor: ",
    ):
        _TransformersModel.from_dict(**{"model": small_vision_model.feature_extractor})


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
    model_provided_card = _fetch_model_card(small_vision_model)
    assert model_provided_card.data.to_dict()["tags"] == ["vision", "image-classification"]
    assert len(model_provided_card.text) > 0


def test_vision_model_save_pipeline_with_defaults(small_vision_model, model_path):
    mlflow.transformers.save_model(transformers_model=small_vision_model, path=model_path)
    # validate inferred pip requirements
    with model_path.joinpath("requirements.txt").open() as file:
        requirements = file.read()
    reqs = {req.split("==")[0] for req in requirements.split("\n")}
    expected_requirements = {"torch", "torchvision", "transformers"}
    assert reqs.intersection(expected_requirements) == expected_requirements
    # validate inferred model card data
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["tags"] == ["vision", "image-classification"]
    # Validate inferred model card text
    with model_path.joinpath("model_card.md").open(encoding="utf-8") as file:
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


def test_vision_model_save__model_for_task_and_card_inference(small_vision_model, model_path):
    mlflow.transformers.save_model(transformers_model=small_vision_model, path=model_path)
    # validate inferred pip requirements
    with model_path.joinpath("requirements.txt").open() as file:
        requirements = file.read()
    reqs = {req.split("==")[0] for req in requirements.split("\n")}
    expected_requirements = {"torch", "torchvision", "transformers"}
    assert reqs.intersection(expected_requirements) == expected_requirements
    # validate inferred model card data
    card_data = yaml.safe_load(model_path.joinpath("model_card_data.yaml").read_bytes())
    assert card_data["tags"] == ["vision", "image-classification"]
    # Validate inferred model card text
    with model_path.joinpath("model_card.md").open(encoding="utf-8") as file:
        card_text = file.read()
    assert len(card_text) > 0

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


def test_component_saving_multi_modal(component_multi_modal, model_path):
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        processor = component_multi_modal["image_processor"]
        expected = {"tokenizer", "processor", "image_processor"}
    else:
        processor = component_multi_modal["feature_extractor"]
        expected = {"tokenizer", "processor", "feature_extractor"}

    mlflow.transformers.save_model(
        transformers_model=component_multi_modal,
        path=model_path,
        processor=processor,
    )
    components_dir = model_path.joinpath("components")
    contents = {item.name for item in components_dir.iterdir()}
    assert contents.intersection(expected) == expected

    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert set(flavor_config["components"]).issubset(expected)


def test_extract_pipeline_components(small_vision_model, small_qa_pipeline):
    components_vision = _record_pipeline_components(small_vision_model)
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        component_list = ["feature_extractor", "image_processor"]
    else:
        component_list = ["feature_extractor"]

    assert components_vision["components"] == component_list
    components_qa = _record_pipeline_components(small_qa_pipeline)
    assert components_qa["tokenizer_type"] == "MobileBertTokenizerFast"
    assert components_qa["components"] == ["tokenizer"]


def test_extract_multi_modal_components(small_multi_modal_pipeline):
    components_multi = _record_pipeline_components(small_multi_modal_pipeline)
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        assert components_multi["image_processor_type"] == "ViltImageProcessor"
        assert components_multi["components"] == ["tokenizer", "image_processor"]
    elif transformers_version >= Version(_IMAGE_PROCESSOR_API_CHANGE_VERSION):
        assert components_multi["feature_extractor_type"] == "ViltFeatureExtractor"
        assert components_multi["components"] == ["feature_extractor", "tokenizer"]
    else:
        assert components_multi["feature_extractor_type"] == "ViltImageProcessor"
        assert components_multi["components"] == ["feature_extractor", "tokenizer"]


def test_basic_save_model_and_load_vision_pipeline(small_vision_model, model_path, image_for_test):
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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

    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        processor_key = "image_processor"
        assert isinstance(loaded_components[processor_key], transformers.ViltImageProcessor)
    else:
        processor_key = "feature_extractor"
        assert isinstance(loaded_components[processor_key], transformers.ViltProcessor)
        assert isinstance(loaded_components["processor"], transformers.ViltProcessor)
    if transformers_version < Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
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


def test_transformers_log_with_duplicate_pip_requirements(small_multi_modal_pipeline, capsys):
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            pip_requirements=["transformers==99.99.99", "transformers", "mlflow"],
        )
    captured = capsys.readouterr()
    assert (
        "Duplicate packages are present within the pip requirements. "
        "Duplicate packages: ['transformers']" in captured.err
    )


def test_transformers_log_with_duplicate_extra_pip_requirements(small_multi_modal_pipeline, capsys):
    with mlflow.start_run():
        mlflow.transformers.log_model(
            small_multi_modal_pipeline,
            "model",
            extra_pip_requirements=["transformers==99.99.99"],
        )
    captured = capsys.readouterr()
    assert (
        "Duplicate packages are present within the pip requirements. "
        "Duplicate packages: ['transformers']" in captured.err
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
        result = mlflow.transformers._fetch_model_card(small_seq2seq_pipeline)

        assert result is None

        mlflow.transformers.save_model(transformers_model=small_seq2seq_pipeline, path=model_path)

        contents = {item.name for item in model_path.iterdir()}
        assert not contents.intersection({"model_card.txt", "model_card_data.yaml"})


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


@flaky()
def test_invalid_task_inference_raises_error(model_path):
    from transformers import Pipeline

    def softmax(outputs):
        maxes = np.max(outputs, axis=-1, keepdims=True)
        shifted_exp = np.exp(outputs - maxes)
        return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

    class PairClassificationPipeline(Pipeline):
        def _sanitize_parameters(self, **kwargs):
            preprocess_kwargs = {}
            if "second_text" in kwargs:
                preprocess_kwargs["second_text"] = kwargs["second_text"]
            return preprocess_kwargs, {}, {}

        # pylint: disable=arguments-renamed,arguments-differ
        def preprocess(self, text, second_text=None):
            return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

        # pylint: disable=arguments-differ,arguments-renamed
        def _forward(self, model_inputs):
            return self.model(**model_inputs)

        # pylint: disable=arguments-differ
        def postprocess(self, model_outputs):
            logits = model_outputs.logits[0].numpy()
            probabilities = softmax(logits)

            best_class = np.argmax(probabilities)
            label = self.model.config.id2label[best_class]
            score = probabilities[best_class].item()
            logits = logits.tolist()
            return {"label": label, "score": score, "logits": logits}

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "sgugger/finetuned-bert-mrpc"
    )
    dummy_pipeline = PairClassificationPipeline(model=model)

    with mock.patch.dict("sys.modules", {"huggingface_hub": None}):
        with pytest.raises(
            MlflowException, match="The task provided is invalid. '' is not a supported"
        ):
            mlflow.transformers.save_model(transformers_model=dummy_pipeline, path=model_path)
        dummy_pipeline.task = "text-classification"
        mlflow.transformers.save_model(transformers_model=dummy_pipeline, path=model_path)


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

    if isinstance(inference_payload, dict):
        pd_input = pd.DataFrame(inference_payload, index=[0])
    else:
        pd_input = pd.DataFrame(inference_payload)
    pd_inference = pyfunc_loaded.predict(pd_input)

    assert isinstance(pd_inference, list)
    assert all(isinstance(element, str) for element in inference)


def read_image(filename):
    image_path = os.path.join(pathlib.Path(__file__).parent.parent, "datasets", filename)
    with open(image_path, "rb") as f:
        return f.read()


def is_base64_image(s):
    try:
        return base64.b64encode(base64.b64decode(s)).decode("utf-8") == s
    except Exception:
        return False


@pytest.mark.parametrize(
    "inference_payload",
    [
        image_url,
        os.path.join(pathlib.Path(__file__).parent.parent, "datasets", "cat.png"),
        "base64",
        Image.open(os.path.join(pathlib.Path(__file__).parent.parent, "datasets", "cat.png")),
    ],
)
def test_vision_pipeline_pyfunc_load_and_infer(small_vision_model, model_path, inference_payload):
    if inference_payload == "base64":
        if Version(transformers.__version__) < Version("4.29"):
            return
        inference_payload = base64.b64encode(read_image("cat_image.jpg")).decode("utf-8")
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
    assert len(predictions) != 0


@pytest.mark.skipif(RUNNING_IN_GITHUB_ACTIONS, reason=GITHUB_ACTIONS_SKIP_REASON)
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
    text2text_generation_pipeline, tmp_path
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

    model_path = tmp_path / "model"
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


def test_text2text_generation_pipeline_with_params_success(text2text_generation_pipeline, tmp_path):
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

    model_path = tmp_path / "model"
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
    text2text_generation_pipeline, tmp_path
):
    data = "muppet keyboard type"
    parameters = {"top_k": 2, "num_beams": 5, "invalid_param": "invalid_param", "do_sample": True}
    generated_output = mlflow.transformers.generate_signature_output(
        text2text_generation_pipeline, data
    )

    model_path = tmp_path / "model"
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

    if all(isinstance(value, str) for value in data.values()):
        pd_input = pd.DataFrame(data, index=[0])
    else:
        pd_input = pd.DataFrame(data)
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
@pytest.mark.skipcacheclean
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


@pytest.mark.parametrize(
    ("pipeline_name", "example", "in_signature", "out_signature"),
    [
        (
            "fill_mask_pipeline",
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
            [{"type": "string"}],
            [{"type": "string"}],
        ),
        (
            "zero_shot_pipeline",
            {
                "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            },
            [
                {"name": "sequences", "type": "string"},
                {"name": "candidate_labels", "type": "string"},
                {"name": "hypothesis_template", "type": "string"},
            ],
            [
                {"name": "sequence", "type": "string"},
                {"name": "labels", "type": "string"},
                {"name": "scores", "type": "double"},
            ],
        ),
    ],
)
@pytest.mark.parametrize("provide_example", [True, False])
@pytest.mark.skipcacheclean
def test_infer_signature_from_example_only(
    pipeline_name, model_path, example, request, provide_example, in_signature, out_signature
):
    pipeline = request.getfixturevalue(pipeline_name)

    input_example = example if provide_example else None
    mlflow.transformers.save_model(pipeline, model_path, input_example=input_example)

    model = Model.load(model_path)

    assert model.signature.inputs.to_dict() == in_signature
    assert model.signature.outputs.to_dict() == out_signature

    if provide_example:
        saved_example = _read_example(model, model_path).to_dict(orient="records")
        if isinstance(example, str):
            assert next(iter(saved_example[0].values())) == example
        elif isinstance(example, list):
            assert list(saved_example[0].values()) == example
        else:
            assert set(saved_example[0].keys()).intersection(example.keys()) == set(
                saved_example[0].keys()
            )
        assert model.saved_input_example_info["type"] == "dataframe"
        orient = "split" if pipeline_name == "zero_shot_pipeline" else "values"
        assert model.saved_input_example_info["pandas_orient"] == orient
    else:
        assert model.saved_input_example_info is None


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
    "inference_payload",
    [
        [os.path.join(pathlib.Path(__file__).parent.parent, "datasets", "cat.png")],
        [image_url, image_url],
        "base64",
    ],
)
def test_vision_pipeline_pyfunc_predict(small_vision_model, inference_payload):
    if not isinstance(inference_payload, list) and inference_payload == "base64":
        if transformers.__version__ < "4.29":
            return
        inference_payload = [
            base64.b64encode(read_image("cat_image.jpg")).decode("utf-8"),
        ]
    artifact_path = "image_classification_model"

    # Log the image classification model
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=small_vision_model,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    inference_payload = json.dumps({"inputs": inference_payload})
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(predictions) != 0


def test_classifier_pipeline_pyfunc_predict(text_classification_pipeline):
    artifact_path = "text_classifier_model"
    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=text_classification_pipeline,
            artifact_path=artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    inference_payload = json.dumps(
        {
            "inputs": [
                "I think this sushi might have gone off",
                "That gym smells like feet, hot garbage, and sadness",
                "I love that we have a moon",
                "I 'love' debugging subprocesses",
                'Quote "in" the string',
            ]
        }
    )

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 5

    # Test the alternate TextClassificationPipeline input structure where text_pair is used
    # and ensure that model serving and direct native inference match
    inference_data = [
        {"text": "test1", "text_pair": "pair1"},
        {"text": "test2", "text_pair": "pair2"},
        {"text": "test 'quote", "text_pair": "pair 'quote'"},
    ]
    inference_payload = json.dumps({"inputs": inference_data})
    response = pyfunc_serve_and_score_model(
        model_uri,
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

    # test simple string input
    inference_payload = json.dumps({"inputs": ["testing"]})

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()

    assert len(values.to_dict()) == 2
    assert len(values.to_dict()["score"]) == 1


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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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


@pytest.mark.parametrize(
    ("pipeline_name", "data", "result"),
    [
        (
            "small_qa_pipeline",
            {"question": "Who's house?", "context": "The house is owned by Run."},
            {
                "inputs": '[{"type": "string", "name": "question"}, {"type": "string", '
                '"name": "context"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "zero_shot_pipeline",
            {
                "sequences": "These pipelines are super cool!",
                "candidate_labels": ["interesting", "uninteresting"],
                "hypothesis_template": "This example talks about how pipelines are {}",
            },
            {
                "inputs": '[{"type": "string", "name": "sequences"}, {"type": "string", '
                '"name": "candidate_labels"}, {"type": "string", "name": '
                '"hypothesis_template"}]',
                "outputs": '[{"type": "string", "name": "sequence"}, {"type": "string", '
                '"name": "labels"}, {"type": "double", "name": "scores"}]',
                "params": None,
            },
        ),
        (
            "text_classification_pipeline",
            "We're just going to have to agree to disagree, then.",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string", "name": "label"}, {"type": "double", "name": '
                '"score"}]',
                "params": None,
            },
        ),
        (
            "table_question_answering_pipeline",
            {
                "query": "how many widgets?",
                "table": json.dumps({"units": ["100", "200"], "widgets": ["500", "500"]}),
            },
            {
                "inputs": '[{"type": "string", "name": "query"}, {"type": "string", "name": '
                '"table"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "summarizer_pipeline",
            "If you write enough tests, you can be sure that your code isn't broken.",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "translation_pipeline",
            "No, I am your father.",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "text_generation_pipeline",
            ["models are", "apples are"],
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "text2text_generation_pipeline",
            ["man apple pie", "dog pizza eat"],
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "fill_mask_pipeline",
            "Juggling <mask> is remarkably dangerous",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "conversational_pipeline",
            "What's shaking, my robot homie?",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
        (
            "ner_pipeline",
            "Blue apples are not a thing",
            {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
                "params": None,
            },
        ),
    ],
)
@pytest.mark.skipcacheclean
def test_signature_inference(pipeline_name, data, result, request):
    pipeline = request.getfixturevalue(pipeline_name)

    default_signature = mlflow.transformers._get_default_pipeline_signature(pipeline)

    assert default_signature.to_dict() == result

    signature_with_input = mlflow.transformers._get_default_pipeline_signature(pipeline, data)

    assert signature_with_input.to_dict() == result


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.skipcacheclean
@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.26.1"), reason="Feature does not exist"
)
@flaky()
def test_extraction_of_torch_dtype_from_pipeline(dtype):
    pipe = transformers.pipeline(
        task="translation_en_to_fr",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
        framework="pt",
        torch_dtype=dtype,
    )

    parsed = mlflow.transformers._extract_torch_dtype_if_set(pipe)

    assert parsed == str(dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32, torch.int64]
)
@pytest.mark.skipcacheclean
def test_deserialization_of_configuration_torch_dtype_entry(dtype):
    flavor_config = {"torch_dtype": str(dtype), "framework": "pt"}

    parsed = mlflow.transformers._deserialize_torch_dtype_if_exists(flavor_config)
    assert isinstance(parsed, torch.dtype)
    assert parsed == dtype


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float64, torch.float, torch.cfloat]
)
@pytest.mark.skipcacheclean
@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.26.1"), reason="Feature does not exist"
)
@flaky()
def test_extraction_of_base_flavor_config(dtype):
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
        torch_dtype=dtype,
        device_map="auto",
        use_auth_token=True,
        trust_remote_code=True,
        revision="main",
        use_fast=True,
    )

    parsed = mlflow.transformers._generate_base_flavor_configuration(full_config_pipeline, task)

    assert parsed == {
        "task": "translation_en_to_fr",
        "instance_type": "TranslationPipeline",
        "source_model_name": "t5-small",
        "pipeline_model_type": "T5ForConditionalGeneration",
        "framework": "pt",
        "torch_dtype": str(dtype),
    }


@pytest.mark.skipcacheclean
@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.26.1"), reason="Feature does not exist"
)
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
@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.26.1"), reason="Feature does not exist"
)
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
@pytest.mark.skipcacheclean
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


@pytest.mark.skipcacheclean
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


@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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
    assert response_data["message"].startswith("Failed to process the input audio data. Either")


@pytest.mark.skipcacheclean
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

    image_file_paths = [image_url]
    parameters = {
        "top_k": 2,
    }
    inference_payload = json.dumps(
        {
            "inputs": image_file_paths,
            "params": parameters,
        }
    )

    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=small_vision_model,
            artifact_path=artifact_path,
            signature=infer_signature(
                image_file_paths,
                mlflow.transformers.generate_signature_output(small_vision_model, image_file_paths),
                params=parameters,
            ),
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )

    predictions = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
    assert len(predictions) == len(image_file_paths)
    assert len(predictions.iloc[0]) == parameters["top_k"]


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
    signature_with_params = infer_signature(
        data,
        mlflow.transformers.generate_signature_output(small_qa_pipeline, data),
        parameters,
    )

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
@pytest.mark.skipcacheclean
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
@pytest.mark.skipcacheclean
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


def test_basic_model_with_accelerate_device_mapping_fails_save(tmp_path):
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
        mlflow.transformers.save_model(transformers_model=pipeline, path=str(tmp_path / "model"))


def test_basic_model_with_accelerate_homogeneous_mapping_works(tmp_path):
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

    mlflow.transformers.save_model(transformers_model=pipeline, path=str(tmp_path / "model"))

    loaded = mlflow.transformers.load_model(str(tmp_path / "model"))
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
    other_files = ["model_card.md", "model_card_data.yaml"]
    for file in other_files:
        path = tmp_path.joinpath(file)
        expected_size += _calculate_expected_size(path)

    mlmodel = yaml.safe_load(tmp_path.joinpath("MLmodel").read_bytes())
    assert mlmodel["model_size_bytes"] == expected_size

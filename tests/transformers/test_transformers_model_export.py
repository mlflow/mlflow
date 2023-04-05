import os

import pandas as pd
from packaging.version import Version
import pathlib
import pytest
import textwrap
from unittest import mock
import yaml

import transformers
from huggingface_hub import ModelCard
from datasets import load_dataset

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.signature import infer_signature
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.transformers import (
    _build_pipeline_from_model_input,
    get_default_pip_requirements,
    get_default_conda_env,
    _infer_transformers_task_type,
    _validate_transformers_task_type,
    _get_instance_type,
    _generate_base_flavor_configuration,
    _TASK_KEY,
    _PIPELINE_MODEL_TYPE_KEY,
    _INSTANCE_TYPE_KEY,
    _MODEL_PATH_OR_NAME_KEY,
    _fetch_model_card,
    _get_base_model_architecture,
    _get_or_infer_task_type,
    _record_pipeline_components,
    _should_add_pyfunc_to_model,
    _TransformersModel,
)
from mlflow.utils.environment import _mlflow_conda_env

from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
    pyfunc_serve_and_score_model,
)

pytestmark = pytest.mark.large

transformers_version = Version(transformers.__version__)
_FEATURE_EXTRACTION_API_CHANGE_VERSION = "4.27.0"
_IMAGE_PROCESSOR_API_CHANGE_VERSION = "4.26.0"


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


@pytest.fixture
def transformers_custom_env(tmp_path):
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["transformers"])
    return conda_env


@pytest.fixture(scope="module")
def small_seq2seq_pipeline():
    # The return type of this model's language head is a List[Dict[str, Any]]
    architecture = "lordtt13/emo-mobilebert"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.TFMobileBertForSequenceClassification.from_pretrained(architecture)
    return transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@pytest.fixture(scope="module")
def small_qa_pipeline():
    # The return type of this model's language head is a Dict[str, Any]
    architecture = "csarron/mobilebert-uncased-squad-v2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.MobileBertForQuestionAnswering.from_pretrained(architecture)
    return transformers.pipeline(task="question-answering", model=model, tokenizer=tokenizer)


@pytest.fixture(scope="module")
def small_vision_model():
    architecture = "google/mobilenet_v2_1.0_224"
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(architecture)
    model = transformers.MobileNetV2ForImageClassification.from_pretrained(architecture)
    return transformers.pipeline(
        task="image-classification", model=model, feature_extractor=feature_extractor
    )


@pytest.fixture(scope="module")
def small_multi_modal_pipeline():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    return transformers.pipeline(model=architecture)


@pytest.fixture(scope="module")
def component_multi_modal():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    tokenizer = transformers.BertTokenizerFast.from_pretrained(architecture)
    processor = transformers.ViltProcessor.from_pretrained(architecture)
    image_processor = transformers.ViltImageProcessor.from_pretrained(architecture)
    model = transformers.ViltForQuestionAnswering.from_pretrained(architecture)
    transformers_model = {"model": model, "tokenizer": tokenizer}
    if transformers_version >= Version(_FEATURE_EXTRACTION_API_CHANGE_VERSION):
        transformers_model["image_processor"] = image_processor
    else:
        transformers_model["feature_extractor"] = processor
    return transformers_model


@pytest.fixture(scope="module")
def small_conversational_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = transformers.AutoModelWithLMHead.from_pretrained("satvikag/chatbot")
    return transformers.pipeline(task="conversational", model=model, tokenizer=tokenizer)


@pytest.fixture(scope="module")
def fill_mask_pipeline():
    return transformers.pipeline(task="fill-mask", model="bert-base-uncased")


@pytest.fixture(scope="module")
def text2text_generation_pipeline():
    task = "text2text-generation"
    architecture = "mrm8488/t5-base-finetuned-question-generation-ap"

    return transformers.pipeline(
        task=task,
        tokenizer=transformers.AutoTokenizer.from_pretrained(architecture),
        model=architecture,
    )


@pytest.fixture(scope="module")
def text_generation_pipeline():
    task = "text-generation"
    architecture = "gpt2"
    return transformers.pipeline(
        task=task,
        model=architecture,
        tokenizer=transformers.AutoTokenizer.from_pretrained(architecture),
    )


@pytest.fixture(scope="module")
def translation_pipeline():
    return transformers.pipeline(task="translation_en_to_de")


@pytest.fixture(scope="module")
def text_classification_pipeline():
    return transformers.pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")


@pytest.fixture(scope="module")
def summarizer_pipeline():
    return transformers.pipeline("summarization")


@pytest.fixture(scope="module")
def zero_shot_pipeline():
    return transformers.pipeline(model="facebook/bart-large-mnli")


@pytest.fixture(scope="module")
def table_question_answering_pipeline():
    return transformers.pipeline(
        task="table-question-answering", model="microsoft/tapex-base-finetuned-wtq"
    )


@pytest.fixture(scope="module")
def ner_pipeline():
    return transformers.pipeline(
        task="token-classification", model="vblagoje/bert-english-uncased-finetuned-pos"
    )


@pytest.fixture(scope="module")
def conversational_pipeline():
    return transformers.pipeline(model="microsoft/DialoGPT-medium")


@pytest.fixture(scope="module")
def image_for_test():
    dataset = load_dataset("huggingface/cats-image")
    return dataset["test"]["image"][0]


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
    ["model", "result"],
    [
        ("small_qa_pipeline", True),
        ("small_seq2seq_pipeline", True),
        ("small_multi_modal_pipeline", False),
        ("small_vision_model", False),
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
    }
    expected_qa_pipeline_conf = {
        _TASK_KEY: "question-answering",
        _INSTANCE_TYPE_KEY: "QuestionAnsweringPipeline",
        _PIPELINE_MODEL_TYPE_KEY: "MobileBertForQuestionAnswering",
        _MODEL_PATH_OR_NAME_KEY: "csarron/mobilebert-uncased-squad-v2",
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


def test_pipeline_saved_model_with_processor_cannot_be_loaded_as_pipeline(
    component_multi_modal, model_path, image_for_test
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
    register_model_patch = mock.patch("mlflow.register_model")
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
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "Question-Answering Model 1",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
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
    registered_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), registered_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "transformers"])
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        # Vision models can't be loaded as pyfunc currently.
        assert pyfunc.FLAVOR_NAME not in model_config.flavors


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


def test_transformers_model_save_without_conda_env_uses_default_env_with_expected_dependencies(
    small_seq2seq_pipeline, model_path
):
    mlflow.transformers.save_model(small_seq2seq_pipeline, model_path)
    _assert_pip_requirements(
        model_path, mlflow.transformers.get_default_pip_requirements(small_seq2seq_pipeline.model)
    )


def test_transformers_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    small_seq2seq_pipeline,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.transformers.log_model(small_seq2seq_pipeline, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(
        model_uri, mlflow.transformers.get_default_pip_requirements(small_seq2seq_pipeline.model)
    )


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


def test_invalid_model_type_without_registered_name_does_not_save(model_path):
    invalid_pipeline = transformers.pipeline(task="text-generation", model="gpt2")
    del invalid_pipeline.model.name_or_path

    with pytest.raises(MlflowException, match="The submitted model type"):
        mlflow.transformers.save_model(transformers_model=invalid_pipeline, path=model_path)


def test_invalid_task_inference_raises_error(model_path):
    import numpy as np

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


@pytest.mark.parametrize(
    "inference_payload, answer",
    [
        (["question: Who's house? context: The house is owned by a man named Run."], "Run"),
        ({"question": "Who's house?", "context": "The house is owned by a man named Run."}, "Run"),
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
            ],
            ["blue", "round and round"],
        ),
        (
            [
                "question: What color is it? context: Some people said it was green but "
                "I know that it's pink.",
                "context: The people on the bus go up and down. Up and down. "
                "question: How do the people go?",
            ],
            ["pink", "up and down"],
        ),
    ],
)
def test_qa_pipeline_pyfunc_load_and_infer(
    small_qa_pipeline, model_path, inference_payload, answer
):
    mlflow.transformers.save_model(
        transformers_model=small_qa_pipeline,
        path=model_path,
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(inference_payload)

    assert inference == answer


@pytest.mark.parametrize(
    "input, result",
    [
        (
            [
                (
                    "answer: To provide high quality answers. context: The best goal "
                    "of any benevolent generative text AI is to answer questions well."
                ),
                (
                    "answer: To distribute electricity. context: Substations exist for "
                    "the sole purpose of routing electrical power to customers from "
                    "the site of generation."
                ),
            ],
            [
                "question: What is the best goal of a benevolent generative text AI?",
                "question: Substations exist for what purpose?",
            ],
        ),
        (
            "answer: It's Run's house. context: The house was purchased by Run "
            "after his initial musical success.",
            "question: What was the name of Run's house?",
        ),
    ],
)
def test_text2text_generation_pipeline_with_inference_configs(
    text2text_generation_pipeline, model_path, input, result
):
    inference_config = {
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    mlflow.transformers.save_model(
        text2text_generation_pipeline, path=model_path, inference_config=inference_config
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(input)

    assert inference == result


@pytest.mark.parametrize(
    "input", [("Generative models are"), (["Generative models are", "Computers are"])]
)
def test_text_generation_pipeline(text_generation_pipeline, model_path, input):
    inference_config = {
        "prefix": "software",
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
    }
    mlflow.transformers.save_model(
        text_generation_pipeline, path=model_path, inference_config=inference_config
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(input)
    if isinstance(input, list):
        assert inference[0].startswith(input[0])
        assert inference[1].startswith(input[1])
    else:
        assert inference.startswith(input)


@pytest.mark.parametrize(
    "input, result",
    [
        ("Riding a [MASK] on the beach is fun!", "bike"),
        (["If I had [MASK], I would fly to the top of a mountain"], ["wings"]),
        (
            ["I use stacks of [MASK] to buy things", "I [MASK] the whole bowl of cherries"],
            ["cash", "ate"],
        ),
    ],
)
def test_fill_mask_pipeline(fill_mask_pipeline, model_path, input, result):
    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(input)
    assert inference == result


def test_pandas_df_input_to_pyfunc(fill_mask_pipeline, model_path):
    mlflow.transformers.save_model(fill_mask_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    mask_input = [
        "Riding a [MASK] on the beach is fun.",
        "If I had [MASK], I would fly to the top of a mountain.",
        "The [MASK] howled at the moon as a pack",
    ]
    df_input = pd.DataFrame([{"input": x} for x in mask_input])

    inference = pyfunc_loaded.predict(df_input)

    assert inference == ["bike", "wings", "wolves"]


@pytest.mark.parametrize(
    "input, result",
    [
        ("I've got a lovely bunch of coconuts!", "Ich habe einen schönen Haufen Kokosnuss!"),
        (
            [
                "I am the very model of a modern major general",
                "Once upon a time, there was a little turtle",
            ],
            [
                "Ich bin das Vorbild eines modernen Großgenerals.",
                "Es gab einmal eine kleine Schildkröte",
            ],
        ),
    ],
)
def test_translation_pipeline(translation_pipeline, model_path, input, result):
    mlflow.transformers.save_model(translation_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(input)
    assert inference == result


@pytest.mark.parametrize(
    "input, result",
    [
        ("I'm telling you that Han shot first!", "POSITIVE"),
        (
            [
                "I think this sushi might have gone off",
                "That gym smells like feet, hot garbage, and sadness",
                "I love that we have a moon",
            ],
            ["NEGATIVE", "NEGATIVE", "POSITIVE"],
        ),
    ],
)
def test_classifier_pipeline(text_classification_pipeline, model_path, input, result):
    mlflow.transformers.save_model(text_classification_pipeline, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    inference = pyfunc_loaded.predict(input)
    assert inference == result


@pytest.mark.parametrize(
    "input",
    ["There once was a boy", ["Baking cookies is quite easy", "Writing unittests is good for"]],
)
def test_summarization_pipeline(summarizer_pipeline, model_path, input):
    inference_config = {
        "prefix": "software",
        "top_k": 2,
        "num_beams": 5,
        "max_length": 30,
        "temperature": 0.62,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "min_length": 10,
    }
    mlflow.transformers.save_model(
        summarizer_pipeline, path=model_path, inference_config=inference_config
    )
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

    inference = pyfunc_loaded.predict(input)
    if isinstance(input, list):
        assert inference[0].strip().startswith(input[0])
        assert inference[1].strip().startswith(input[1])
    else:
        assert inference.strip().startswith(input)


@pytest.mark.parametrize(
    "input, result",
    [
        (
            {
                "sequences": "I love the latest update to this IDE!",
                "candidate_labels": ["happy", "sad"],
            },
            "happy",
        ),
        (
            {
                "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            },
            ["happy", "sad"],
        ),
    ],
)
def test_zero_shot_classification_pipeline(zero_shot_pipeline, model_path, input, result):
    mlflow.transformers.save_model(zero_shot_pipeline, model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict(input)

    assert inference == result


def test_table_question_answering_pipeline(table_question_answering_pipeline, model_path):
    table = {
        "Fruit": ["Apples", "Bananas", "Oranges", "Watermelon", "Blueberries"],
        "Sales": ["1230945.55", "86453.12", "11459.23", "8341.23", "2325.88"],
        "Inventory": ["910", "4589", "11200", "80", "3459"],
    }
    mlflow.transformers.save_model(table_question_answering_pipeline, model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict({"query": "What should we order more of?", "table": table})
    assert inference == "apples"

    inference_multiple = loaded_pyfunc.predict(
        {"query": ["What is our highest sales?", "What should we order more of?"], "table": table}
    )
    assert inference_multiple == ["1230945.55", "apples"]


def test_conversational_pipeline(conversational_pipeline, model_path):
    mlflow.transformers.save_model(conversational_pipeline, model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)

    first_response = loaded_pyfunc.predict("What is the best way to get to Antarctica?")

    assert first_response == "I think you can get there by boat."

    second_response = loaded_pyfunc.predict("What kind of boat should I use?")

    assert second_response == "A boat that can go to Antarctica."

    # Test that a new loaded instance has no context.
    loaded_again_pyfunc = mlflow.pyfunc.load_model(model_path)
    third_response = loaded_again_pyfunc.predict("What kind of boat should I use?")

    assert third_response == "A boat that can't sink."

    fourth_response = loaded_again_pyfunc.predict("Can I use it to go to the moon?")

    assert fourth_response == "Only if you have a boat that can't sink."


@pytest.mark.parametrize(
    "input, in_signature, out_signature",
    [
        ("Generative models are", '[{"type": "string"}]', '[{"type": "string"}]'),
        (
            ["Computers are", "Dogs are"],
            '[{"type": "string"}, {"type": "string"}]',
            '[{"type": "string"}, {"type": "string"}]',
        ),
    ],
)
def test_signature_inference_strings_and_lists_of_strings(
    text_generation_pipeline, model_path, input, in_signature, out_signature
):
    mlflow.transformers.save_model(text_generation_pipeline, model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)

    answer = loaded_pyfunc.predict(input)
    signature = infer_signature(input, answer)

    assert signature.inputs.to_json() == in_signature
    assert signature.outputs.to_json() == out_signature


@pytest.mark.parametrize(
    "input, result",
    [
        (
            "I have a dog and his name is Willy!",
            "PRON,VERB,DET,NOUN,CCONJ,PRON,NOUN,AUX,PROPN,PUNCT",
        ),
        (
            ["We are the knights who say nee!", "Houston, we may have a problem."],
            [
                "PRON,AUX,DET,PROPN,PRON,VERB,INTJ,PUNCT",
                "PROPN,PUNCT,PRON,AUX,VERB,DET,NOUN,PUNCT",
            ],
        ),
    ],
)
def test_ner_pipeline(ner_pipeline, model_path, input, result):
    mlflow.transformers.save_model(ner_pipeline, model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_path)
    inference = loaded_pyfunc.predict(input)

    assert inference == result


def test_signature_inference_dict_input_to_string_output():
    pass


# TODO: make sure that _enforce_schema can support input of scalar str, list[str], dict[str, dict]


def test_model_saved_with_signature_using_string_input():
    pass


# TODO: ensure that


def test_model_saved_with_signature_using_list_of_string_input():
    pass


def test_model_saved_with_signature_using_dict_input():
    pass


def test_model_saved_with_signature_using_compound_input():
    pass


#
# def test_qa_pipeline_pyfunc_predict(small_qa_pipeline, tmp_path):
#     artifact_path = "qa_model"
#     with mlflow.start_run():
#         mlflow.transformers.log_model(
#             transformers_model=small_qa_pipeline,
#             artifact_path=artifact_path,
#         )
#         model_uri = mlflow.get_artifact_uri(artifact_path)
#
#     inference_payload = ["question: Who's house? context: The house is owned by a man named Run."]
#
#     from mlflow.deployments import PredictionsResponse
#
#     response = pyfunc_serve_and_score_model(
#         model_uri,
#         data=inference_payload,
#         content_type=pyfunc_scoring_server.CONTENT_TYPE_CSV,
#     )
#     values = PredictionsResponse.from_json(response.content.decode("utf-8")).get_predictions()
#
#     print(values)
#
#     assert 1 == 0


def test_loading_unsupported_pipeline_type_as_pyfunc(small_multi_modal_pipeline, model_path):
    mlflow.transformers.save_model(small_multi_modal_pipeline, model_path)
    with pytest.raises(MlflowException, match='Model does not have the "python_function" flavor'):
        mlflow.pyfunc.load_model(model_path)

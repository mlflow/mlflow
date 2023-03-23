import os
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
    _LOGGED_TYPE_KEY,
    _INSTANCE_TYPE_KEY,
    _MODEL_PATH_OR_NAME_KEY,
    _fetch_model_card,
    _get_base_model_architecture,
    _get_or_infer_task_type,
    _record_pipeline_components,
)
from mlflow.utils.environment import _mlflow_conda_env

from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
)

pytestmark = pytest.mark.large


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
    model = transformers.ViltForQuestionAnswering.from_pretrained(architecture)
    return processor, model, tokenizer


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
    assert _infer_transformers_task_type(small_seq2seq_pipeline.model) == expected_task
    with pytest.raises(MlflowException, match="The provided model type"):
        _infer_transformers_task_type(small_seq2seq_pipeline.tokenizer)


def test_task_validation():
    with pytest.raises(MlflowException, match="The task provided is invalid. 'fake-task' is not"):
        _validate_transformers_task_type("fake-task")
    _validate_transformers_task_type("image-classification")


def test_instance_extraction(small_qa_pipeline):
    assert _get_instance_type(small_qa_pipeline, True) == "Pipeline"
    assert _get_instance_type(small_qa_pipeline, False) == "QuestionAnsweringPipeline"
    assert _get_instance_type(small_qa_pipeline.model, True) == "PreTrainedModel"
    assert _get_instance_type(small_qa_pipeline.model, False) == "MobileBertForQuestionAnswering"


def test_model_architecture_extraction(small_seq2seq_pipeline):
    assert _get_base_model_architecture(small_seq2seq_pipeline) == "lordtt13/emo-mobilebert"
    assert _get_base_model_architecture((small_seq2seq_pipeline.model)) == "lordtt13/emo-mobilebert"


def test_base_flavor_configuration_generation(small_seq2seq_pipeline, small_qa_pipeline):
    expected_seq_pipeline_conf = {
        _TASK_KEY: "text-classification",
        _LOGGED_TYPE_KEY: "Pipeline",
        _INSTANCE_TYPE_KEY: "TextClassificationPipeline",
        _PIPELINE_MODEL_TYPE_KEY: "TFMobileBertForSequenceClassification",
        _MODEL_PATH_OR_NAME_KEY: "lordtt13/emo-mobilebert",
    }
    expected_seq_model_conf = {
        _TASK_KEY: "text-classification",
        _LOGGED_TYPE_KEY: "TFPreTrainedModel",
        _INSTANCE_TYPE_KEY: "TFMobileBertForSequenceClassification",
        _MODEL_PATH_OR_NAME_KEY: "lordtt13/emo-mobilebert",
    }

    expected_qa_pipeline_conf = {
        _TASK_KEY: "question-answering",
        _LOGGED_TYPE_KEY: "Pipeline",
        _INSTANCE_TYPE_KEY: "QuestionAnsweringPipeline",
        _PIPELINE_MODEL_TYPE_KEY: "MobileBertForQuestionAnswering",
        _MODEL_PATH_OR_NAME_KEY: "csarron/mobilebert-uncased-squad-v2",
    }
    expected_qa_model_conf = {
        _TASK_KEY: "question-answering",
        _LOGGED_TYPE_KEY: "PreTrainedModel",
        _INSTANCE_TYPE_KEY: "MobileBertForQuestionAnswering",
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
    seq_conf_infer_model = _generate_base_flavor_configuration(
        small_seq2seq_pipeline.model, _get_or_infer_task_type(small_seq2seq_pipeline.model)
    )
    assert seq_conf_infer_model == expected_seq_model_conf
    qa_conf_infer_task = _generate_base_flavor_configuration(
        small_qa_pipeline, _get_or_infer_task_type(small_qa_pipeline)
    )
    assert qa_conf_infer_task == expected_qa_pipeline_conf
    qa_conf_specify_task = _generate_base_flavor_configuration(
        small_qa_pipeline, "question-answering"
    )
    assert qa_conf_specify_task == expected_qa_pipeline_conf
    qa_conf_infer_model = _generate_base_flavor_configuration(
        small_qa_pipeline.model, _get_or_infer_task_type(small_qa_pipeline.model)
    )
    assert qa_conf_infer_model == expected_qa_model_conf
    with pytest.raises(MlflowException, match="The task provided is invalid. 'magic' is not"):
        _generate_base_flavor_configuration(small_qa_pipeline, "magic")


def test_pipeline_construction_from_base_nlp_model(small_qa_pipeline):
    generated = _build_pipeline_from_model_input(small_qa_pipeline.model, "question-answering")
    assert isinstance(generated, type(small_qa_pipeline))
    assert isinstance(generated.tokenizer, type(small_qa_pipeline.tokenizer))


def test_pipeline_construction_from_base_vision_model(small_vision_model):
    generated = _build_pipeline_from_model_input(small_vision_model.model, "image-classification")
    assert isinstance(generated, type(small_vision_model))
    assert isinstance(generated.tokenizer, type(small_vision_model.tokenizer))
    # The type is equivalent below, having the exact same properties and behavior between
    # the (higher level API) ImageProcessor vs. the lower-level FeatureExtractor
    assert isinstance(generated.feature_extractor, transformers.MobileNetV2ImageProcessor)


def test_pipeline_construction_fails_with_invalid_type(small_vision_model):
    with pytest.raises(MlflowException, match="The provided model is not the correct type. The"):
        _build_pipeline_from_model_input(
            small_vision_model.feature_extractor, "image_classification"
        )


def test_model_card_acquisition_vision_model(small_vision_model):
    model_provided_card = _fetch_model_card(small_vision_model)
    assert model_provided_card.data.to_dict()["tags"] == ["vision", "image-classification"]
    assert model_provided_card.text.startswith(
        "\n# MobileNet V2\n\nMobileNet V2 model pre-trained on ImageNet-1k"
    )


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
    with model_path.joinpath("model_card_text.txt").open() as file:
        card_text = file.read()
    assert card_text.startswith("\n# MobileNet V2\n\nMobileNet V2 model pre-trained on ImageNet-1k")
    # Validate conda.yaml
    conda_env = yaml.safe_load(model_path.joinpath("conda.yaml").read_bytes())
    assert {req.split("==")[0] for req in conda_env["dependencies"][2]["pip"]}.intersection(
        expected_requirements
    ) == expected_requirements
    # Validate the MLModel file
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "ImageClassificationPipeline"
    assert flavor_config["logged_type"] == "Pipeline"
    assert flavor_config["pipeline_model_type"] == "MobileNetV2ForImageClassification"
    assert flavor_config["task"] == "image-classification"
    assert flavor_config["source_model_name"] == "google/mobilenet_v2_1.0_224"


def test_qa_model_save_model_for_task_and_card_inference(small_seq2seq_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=small_seq2seq_pipeline.model,
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
    with model_path.joinpath("model_card_text.txt").open() as file:
        card_text = file.read()
    assert card_text.startswith("\n## Emo-MobileBERT: a thin version of BERT LARGE, trained on")
    # validate MLmodel files
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "TextClassificationPipeline"
    assert flavor_config["logged_type"] == "Pipeline"
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
    with model_path.joinpath("model_card_text.txt").open() as file:
        card_text = file.read()
    assert card_text.startswith("\n# I made a new model!")
    # validate MLmodel files
    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert flavor_config["instance_type"] == "QuestionAnsweringPipeline"
    assert flavor_config["logged_type"] == "Pipeline"
    assert flavor_config["pipeline_model_type"] == "MobileBertForQuestionAnswering"
    assert flavor_config["task"] == "question-answering"
    assert flavor_config["source_model_name"] == "csarron/mobilebert-uncased-squad-v2"


def test_basic_save_model_and_load_text_pipeline(small_seq2seq_pipeline, model_path):
    mlflow.transformers.save_model(
        transformers_model=small_seq2seq_pipeline.model,
        path=model_path,
    )
    loaded = mlflow.transformers.load_model(model_path)
    result = loaded("MLflow is a really neat tool!")
    assert result[0]["label"] == "happy"
    assert result[0]["score"] > 0.5


def test_component_saving_multi_modal(component_multi_modal, model_path):
    processor, model, _ = component_multi_modal
    mlflow.transformers.save_model(transformers_model=model, path=model_path, processor=processor)
    expected = {"tokenizer", "processor", "image_processor", "feature_extractor"}
    components_dir = model_path.joinpath("components")
    contents = {item.name for item in components_dir.iterdir()}
    assert contents.intersection(expected) == expected

    mlmodel = yaml.safe_load(model_path.joinpath("MLmodel").read_bytes())
    flavor_config = mlmodel["flavors"]["transformers"]
    assert set(flavor_config["components"]).issubset(expected)


def test_extract_pipeline_components(small_vision_model, small_qa_pipeline):
    components_vision = _record_pipeline_components(small_vision_model)
    assert components_vision["feature_extractor_type"] == "MobileNetV2FeatureExtractor"
    assert components_vision["components"] == ["feature_extractor", "image_processor"]
    components_qa = _record_pipeline_components(small_qa_pipeline)
    assert components_qa["tokenizer_type"] == "MobileBertTokenizerFast"
    assert components_qa["components"] == ["tokenizer"]


def test_extract_multi_modal_components(small_multi_modal_pipeline):
    components_multi = _record_pipeline_components(small_multi_modal_pipeline)
    assert components_multi["image_processor_type"] == "ViltImageProcessor"
    assert components_multi["components"] == ["tokenizer", "image_processor"]


def test_basic_save_model_and_load_vision_pipeline(small_vision_model, model_path, image_for_test):
    mlflow.transformers.save_model(
        transformers_model=small_vision_model.model,
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
    expected_components = {"model", "task", "tokenizer", "image_processor"}
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
    processor, model, _ = component_multi_modal
    mlflow.transformers.save_model(transformers_model=model, path=model_path, processor=processor)
    # Ensure that the appropriate Processor object was detected and loaded with the pipeline.
    loaded_components = mlflow.transformers.load_model(
        model_uri=model_path, return_type="components"
    )
    assert isinstance(loaded_components["model"], transformers.ViltForQuestionAnswering)
    assert isinstance(loaded_components["tokenizer"], transformers.BertTokenizerFast)
    assert isinstance(loaded_components["processor"], transformers.ViltProcessor)
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


def test_processor_type_model_loaded_as_pipeline_raises(component_multi_modal, model_path):
    processor, model, _ = component_multi_modal
    mlflow.transformers.save_model(transformers_model=model, path=model_path, processor=processor)
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
    artifact_path = "transformers"
    registered_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), registered_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["tensorflow", "transformers"])
        mlflow.transformers.log_model(
            transformers_model=small_vision_model.model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


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

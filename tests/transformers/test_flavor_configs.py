import pytest

from mlflow.exceptions import MlflowException
from mlflow.transformers import _build_pipeline_from_model_input
from mlflow.transformers.flavor_config import (
    build_flavor_config,
    update_flavor_conf_to_persist_pretrained_model,
)
from mlflow.transformers.hub_utils import is_valid_hf_repo_id

from tests.transformers.helper import IS_NEW_FEATURE_EXTRACTION_API


@pytest.fixture
def multi_modal_pipeline(component_multi_modal):
    task = "image-classification"
    pipeline = _build_pipeline_from_model_input(component_multi_modal, task)

    if IS_NEW_FEATURE_EXTRACTION_API:
        processor = pipeline.image_processor
        components = {
            "tokenizer": "BertTokenizerFast",
            "image_processor": "ViltImageProcessor",
            "processor": "ViltImageProcessor",
        }
    else:
        processor = pipeline.feature_extractor
        components = {
            "tokenizer": "BertTokenizerFast",
            "feature_extractor": "ViltProcessor",
            "processor": "ViltProcessor",
        }

    return pipeline, task, processor, components


def test_flavor_config_tf(small_seq2seq_pipeline):
    expected = {
        "task": "text-classification",
        "instance_type": "TextClassificationPipeline",
        "pipeline_model_type": "TFMobileBertForSequenceClassification",
        "source_model_name": "lordtt13/emo-mobilebert",
        "model_binary": "model",
        "framework": "tf",
        "components": ["tokenizer"],
        "tokenizer_type": "MobileBertTokenizerFast",
    }
    conf = build_flavor_config(small_seq2seq_pipeline)
    assert conf == expected


def test_flavor_config_pt_save_pretrained_false(small_qa_pipeline):
    expected = {
        "task": "question-answering",
        "instance_type": "QuestionAnsweringPipeline",
        "pipeline_model_type": "MobileBertForQuestionAnswering",
        "source_model_name": "csarron/mobilebert-uncased-squad-v2",
        # "source_model_revision": "SOME_COMMIT_SHA",
        "framework": "pt",
        "components": ["tokenizer"],
        "tokenizer_type": "MobileBertTokenizerFast",
        "tokenizer_name": "csarron/mobilebert-uncased-squad-v2",
        # "tokenizer_revision": "SOME_COMMIT_SHA",
    }
    conf = build_flavor_config(small_qa_pipeline, save_pretrained=False)
    assert len(conf.pop("source_model_revision")) == 40
    assert len(conf.pop("tokenizer_revision")) == 40
    assert conf == expected


def test_flavor_config_component_multi_modal(multi_modal_pipeline):
    pipeline, task, processor, expected_components = multi_modal_pipeline

    # 1. Test with save_pretrained = True
    conf = build_flavor_config(pipeline, processor)

    assert "model_binary" in conf
    assert conf["pipeline_model_type"] == "ViltForQuestionAnswering"
    assert conf["source_model_name"] == "dandelin/vilt-b32-finetuned-vqa"
    assert "source_model_revision" not in conf

    assert set(conf["components"]) == set(expected_components.keys()) - {"processor"}
    for component in expected_components:
        assert conf[f"{component}_type"] == expected_components[component]
        assert f"{component}_revision" not in conf
        assert f"{component}_revision" not in conf


def test_flavor_config_component_multi_modal_save_pretrained_false(multi_modal_pipeline):
    pipeline, task, processor, expected_components = multi_modal_pipeline

    conf = build_flavor_config(pipeline, processor, False)

    assert "model_binary" not in conf
    assert conf["pipeline_model_type"] == "ViltForQuestionAnswering"
    assert conf["source_model_name"] == pipeline.model.name_or_path
    assert len(conf["source_model_revision"]) == 40

    assert set(conf["components"]) == set(expected_components.keys()) - {"processor"}

    for component in expected_components:
        assert conf[f"{component}_type"] == expected_components[component]
        assert conf[f"{component}_name"] == pipeline.model.name_or_path
        assert len(conf[f"{component}_revision"]) == 40


def test_is_valid_hf_repo_id(tmp_path):
    assert is_valid_hf_repo_id(None) is False
    assert is_valid_hf_repo_id(str(tmp_path)) is False
    assert is_valid_hf_repo_id("invalid/repo/name") is False
    assert is_valid_hf_repo_id("google-t5/t5-small") is True


_COMMON_CONF = {
    "task": "text-classification",
    "instance_type": "TextClassificationPipeline",
    "pipeline_model_type": "TFMobileBertForSequenceClassification",
    "source_model_name": "lordtt13/emo-mobilebert",
    "framework": "tf",
    "components": ["tokenizer"],
    "tokenizer_type": "MobileBertTokenizerFast",
    "transformers_version": "4.37.1",
}
_COMMIT_HASH = "26d8fcb41762ae83cc9fa03005cb63cde06ef340"


def test_update_flavor_conf_to_persist_pretrained_model():
    flavor_conf = {
        **_COMMON_CONF,
        "components": ["tokenizer"],
        "source_model_revision": _COMMIT_HASH,
        "tokenizer_name": "lordtt13/emo-mobilebert",
        "tokenizer_revision": _COMMIT_HASH,
    }
    updated_flavor_conf = update_flavor_conf_to_persist_pretrained_model(flavor_conf)

    assert updated_flavor_conf["model_binary"] == "model"
    assert "source_model_revision" not in updated_flavor_conf
    assert "tokenizer_revision" not in updated_flavor_conf
    assert "tokenizer_name" not in updated_flavor_conf


def test_update_flavor_conf_to_persist_pretrained_model_multi_modal():
    flavor_conf = {
        **_COMMON_CONF,
        "components": ["tokenizer", "image_processor"],
        "source_model_revision": _COMMIT_HASH,
        "tokznier_revision": _COMMIT_HASH,
        "image-processor_type": "ViltImageProcessor",
        "image_processor_name": "dandelin/vilt-b32-finetuned-vqa",
        "image_processor_revision": _COMMIT_HASH,
        "processor_type": "ViltImageProcessor",
        "processor_name": "dandelin/vilt-b32-finetuned-vqa",
        "processor_revision": _COMMIT_HASH,
    }
    updated_flavor_conf = update_flavor_conf_to_persist_pretrained_model(flavor_conf)

    assert updated_flavor_conf["model_binary"] == "model"
    assert "source_model_revision" not in updated_flavor_conf
    for component in ["tokenizer", "image_processor", "processor"]:
        assert f"{component}_revision" not in updated_flavor_conf
        assert f"{component}_name" not in updated_flavor_conf


def test_update_flavor_conf_to_persist_pretrained_model_raise_if_already_persisted():
    flavor_conf = {
        **_COMMON_CONF,
        "components": ["tokenizer"],
        "model_binary": "model",
    }

    with pytest.raises(MlflowException, match="It appears that the pretrained model weight"):
        update_flavor_conf_to_persist_pretrained_model(flavor_conf)

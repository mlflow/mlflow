import importlib

import pytest

import mlflow
from mlflow.models import Model
from mlflow.transformers.peft import get_peft_base_model, is_peft_model

from tests.transformers.test_transformers_model_export import (
    HF_COMMIT_HASH_PATTERN,
    model_path,  # noqa: F401
)

peft_not_installed_cond = {
    "condition": importlib.util.find_spec("peft") is None,
    "reason": "PEFT is not installed",
}


@pytest.mark.skipif(**peft_not_installed_cond)
def test_is_peft_model(peft_pipeline, small_qa_pipeline):
    assert is_peft_model(peft_pipeline.model)
    assert not is_peft_model(small_qa_pipeline.model)


@pytest.mark.skipif(**peft_not_installed_cond)
def test_get_peft_base_model(peft_pipeline):
    base_model = get_peft_base_model(peft_pipeline.model)
    assert base_model.__class__.__name__ == "OPTForCausalLM"
    assert base_model.name_or_path == "facebook/opt-350m"


@pytest.mark.skipif(**peft_not_installed_cond)
def test_get_peft_base_model_prompt_learning(small_qa_pipeline):
    from peft import PeftModel, PromptTuningConfig

    peft_config = PromptTuningConfig(
        task_type="question-answering",
        num_virtual_tokens=10,
        peft_type="PROMPT_TUNING",
    )
    peft_model = PeftModel(small_qa_pipeline.model, peft_config)

    base_model = get_peft_base_model(peft_model)
    assert base_model.__class__.__name__ == "MobileBertForQuestionAnswering"
    assert base_model.name_or_path == "csarron/mobilebert-uncased-squad-v2"


@pytest.mark.skipif(**peft_not_installed_cond)
def test_save_and_load_peft_pipeline(peft_pipeline, model_path):
    from peft import PeftModel

    mlflow.transformers.save_model(
        transformers_model=peft_pipeline,
        path=model_path,
    )

    # For PEFT, only the adapter model should be saved
    assert model_path.joinpath("peft").exists()
    assert not model_path.joinpath("model").exists()
    assert not model_path.joinpath("components").exists()

    # Validate the contents of MLModel file
    flavor_conf = Model.load(str(model_path.joinpath("MLmodel"))).flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert HF_COMMIT_HASH_PATTERN.match(flavor_conf["source_model_revision"])
    assert flavor_conf["peft_adaptor"] == "peft"

    loaded_pipeline = mlflow.transformers.load_model(model_path)
    assert isinstance(loaded_pipeline.model, PeftModel)

    loaded_pipeline.predict("Hi")


@pytest.mark.skipif(**peft_not_installed_cond)
def test_save_and_load_peft_components(peft_pipeline, model_path):
    from peft import PeftModel

    mlflow.transformers.save_model(
        transformers_model={
            "model": peft_pipeline.model,
            "tokenizer": peft_pipeline.tokenizer,
        },
        path=model_path,
    )

    loaded_pipeline = mlflow.transformers.load_model(model_path)
    assert isinstance(loaded_pipeline.model, PeftModel)

    loaded_pipeline.predict("Hi")

import importlib

import pytest
import transformers
from packaging.version import Version

import mlflow
from mlflow.models import Model
from mlflow.transformers.peft import get_peft_base_model, is_peft_model

SKIP_IF_PEFT_NOT_AVAILABLE = pytest.mark.skipif(
    (
        importlib.util.find_spec("peft") is None
        or Version(transformers.__version__) <= Version("4.25.1")
    ),
    reason="PEFT is not installed or Transformer version is too old",
)
pytestmark = SKIP_IF_PEFT_NOT_AVAILABLE


def test_is_peft_model(peft_pipeline, small_qa_pipeline):
    assert is_peft_model(peft_pipeline.model)
    assert not is_peft_model(small_qa_pipeline.model)


def test_get_peft_base_model(peft_pipeline):
    base_model = get_peft_base_model(peft_pipeline.model)
    assert base_model.__class__.__name__ == "BertForSequenceClassification"
    assert base_model.name_or_path == "Elron/bleurt-tiny-512"


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


def test_save_and_load_peft_pipeline(peft_pipeline, tmp_path):
    import peft

    from tests.transformers.test_transformers_model_export import HF_COMMIT_HASH_PATTERN

    mlflow.transformers.save_model(
        transformers_model=peft_pipeline,
        path=tmp_path,
    )

    # For PEFT, only the adapter model should be saved
    assert tmp_path.joinpath("peft").exists()
    assert not tmp_path.joinpath("model").exists()
    assert not tmp_path.joinpath("components").exists()

    # Validate the contents of MLModel file
    flavor_conf = Model.load(str(tmp_path.joinpath("MLmodel"))).flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert HF_COMMIT_HASH_PATTERN.match(flavor_conf["source_model_revision"])
    assert flavor_conf["peft_adaptor"] == "peft"

    # Validate peft is recorded to requirements.txt
    with open(tmp_path.joinpath("requirements.txt")) as f:
        assert f"peft=={peft.__version__}" in f.read()

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, peft.PeftModel)
    loaded_pipeline.predict("Hi")


def test_save_and_load_peft_components(peft_pipeline, tmp_path, capsys):
    from peft import PeftModel

    mlflow.transformers.save_model(
        transformers_model={
            "model": peft_pipeline.model,
            "tokenizer": peft_pipeline.tokenizer,
        },
        path=tmp_path,
    )

    # PEFT pipeline construction error should not be raised
    peft_err_msg = (
        "The model 'PeftModelForSequenceClassification' is not supported for text-classification"
    )
    assert peft_err_msg not in capsys.readouterr().err

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")


def test_log_peft_pipeline(peft_pipeline):
    from peft import PeftModel

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=peft_pipeline,
            artifact_path="model",
            input_example="hi",
        )

    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")

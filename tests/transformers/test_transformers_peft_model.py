import importlib
import os
import re

import pytest
import transformers

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.transformers.flavor_config import FlavorKey
from mlflow.transformers.peft import get_peft_base_model, is_peft_model
from mlflow.utils.logging_utils import suppress_logs

SKIP_IF_PEFT_NOT_AVAILABLE = pytest.mark.skipif(
    importlib.util.find_spec("peft") is None,
    reason="PEFT is not installed",
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
    from peft import PeftModel, PromptTuningConfig, TaskType

    peft_config = PromptTuningConfig(
        task_type=TaskType.QUESTION_ANS,
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
        model_info = mlflow.transformers.log_model(peft_pipeline, name="model", input_example="hi")

    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")


@pytest.fixture
def peft_model_with_local_base(tmp_path_factory):
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_PIPELINE_ERROR_MSG = re.compile(r"is not supported for")

    base_model_id = "Elron/bleurt-tiny-512"
    base_dir = tmp_path_factory.mktemp("base_model")

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id)

    base_model.save_pretrained(str(base_dir))
    tokenizer.save_pretrained(str(base_dir))

    local_model = transformers.AutoModelForSequenceClassification.from_pretrained(str(base_dir))
    local_tokenizer = transformers.AutoTokenizer.from_pretrained(str(base_dir))

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    peft_model = get_peft_model(local_model, peft_config)

    with suppress_logs("transformers.pipelines.base", filter_regex=_PEFT_PIPELINE_ERROR_MSG):
        pipeline = transformers.pipeline(
            task="text-classification", model=peft_model, tokenizer=local_tokenizer
        )

    return pipeline, str(base_dir)


def test_save_and_load_peft_with_base_model_path(peft_model_with_local_base, tmp_path):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base

    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=tmp_path,
        base_model_path=base_dir,
    )

    # PEFT adapter should be saved, components should be saved, but base model should NOT
    assert tmp_path.joinpath("peft").exists()
    assert not tmp_path.joinpath("model").exists()
    assert tmp_path.joinpath("components").exists()

    # Validate flavor config
    flavor_conf = Model.load(str(tmp_path.joinpath("MLmodel"))).flavors["transformers"]
    assert "model_binary" not in flavor_conf
    assert "source_model_revision" not in flavor_conf
    assert flavor_conf[FlavorKey.MODEL_LOCAL_BASE] == os.path.abspath(base_dir)
    assert flavor_conf[FlavorKey.PEFT] == "peft"

    loaded_pipeline = mlflow.transformers.load_model(tmp_path)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")


def test_save_peft_with_base_model_path_components(peft_model_with_local_base, tmp_path):
    pipeline, base_dir = peft_model_with_local_base

    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=tmp_path,
        base_model_path=base_dir,
    )

    components_dir = tmp_path / "components" / "tokenizer"
    assert components_dir.exists()
    assert any(components_dir.iterdir())


def test_log_peft_with_base_model_path(peft_model_with_local_base):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            pipeline,
            name="model",
            base_model_path=base_dir,
            input_example="hi",
        )

    loaded_pipeline = mlflow.transformers.load_model(model_info.model_uri)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")


def test_base_model_path_rejects_non_peft_model(small_qa_pipeline, tmp_path):
    with pytest.raises(MlflowException, match="only supported for PEFT models"):
        mlflow.transformers.save_model(
            transformers_model=small_qa_pipeline,
            path=tmp_path,
            base_model_path="/some/path",
        )


def test_base_model_path_rejects_invalid_path(peft_model_with_local_base, tmp_path):
    pipeline, _ = peft_model_with_local_base

    with pytest.raises(MlflowException, match="does not exist"):
        mlflow.transformers.save_model(
            transformers_model=pipeline,
            path=tmp_path,
            base_model_path="/nonexistent/path/to/model",
        )


def test_load_peft_with_base_model_path_override(peft_model_with_local_base, tmp_path):
    from peft import PeftModel

    pipeline, base_dir = peft_model_with_local_base
    save_dir = tmp_path / "model_output"

    # Save with a dummy path (simulating save on a different machine)
    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=save_dir,
        base_model_path=base_dir,
    )

    # Load with an explicit override path (simulating different mount point)
    loaded_pipeline = mlflow.transformers.load_model(save_dir, base_model_path=base_dir)
    assert isinstance(loaded_pipeline.model, PeftModel)
    loaded_pipeline.predict("Hi")


def test_base_model_path_rejects_non_checkpoint_dir(peft_model_with_local_base, tmp_path):
    pipeline, _ = peft_model_with_local_base

    empty_dir = tmp_path / "empty_base"
    empty_dir.mkdir()

    save_dir = tmp_path / "model_output"
    with pytest.raises(MlflowException, match="config.json"):
        mlflow.transformers.save_model(
            transformers_model=pipeline,
            path=save_dir,
            base_model_path=str(empty_dir),
        )


def test_load_base_model_path_override_rejects_non_checkpoint_dir(
    peft_model_with_local_base, tmp_path
):
    pipeline, base_dir = peft_model_with_local_base
    save_dir = tmp_path / "model_output"
    mlflow.transformers.save_model(
        transformers_model=pipeline,
        path=save_dir,
        base_model_path=base_dir,
    )

    empty_dir = tmp_path / "empty_override"
    empty_dir.mkdir()

    with pytest.raises(MlflowException, match="config.json"):
        mlflow.transformers.load_model(save_dir, base_model_path=str(empty_dir))

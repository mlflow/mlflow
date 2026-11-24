import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("diffusers")
pytest.importorskip("safetensors")

from unittest.mock import MagicMock, Mock, patch

from safetensors.numpy import save_file

import mlflow
import mlflow.diffusers
from mlflow.diffusers import FLAVOR_NAME, DiffusersAdapterModel
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def _create_fake_adapter(tmp_path, filename="adapter.safetensors"):
    adapter_dir = tmp_path / "fake_adapter"
    adapter_dir.mkdir(exist_ok=True)
    tensors = {"lora_weight": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / filename))
    return adapter_dir


@pytest.fixture
def adapter_dir(tmp_path):
    return _create_fake_adapter(tmp_path)


@pytest.fixture
def adapter_file(tmp_path):
    adapter_dir = _create_fake_adapter(tmp_path)
    return adapter_dir / "adapter.safetensors"


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model_output"


def test_save_model_from_directory(adapter_dir, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    mlmodel_path = model_path / MLMODEL_FILE_NAME
    assert mlmodel_path.exists()

    with open(mlmodel_path) as f:
        mlmodel = yaml.safe_load(f)

    assert FLAVOR_NAME in mlmodel["flavors"]
    assert "python_function" in mlmodel["flavors"]

    flavor_conf = mlmodel["flavors"][FLAVOR_NAME]
    assert flavor_conf["base_model"] == BASE_MODEL_ID
    assert flavor_conf["adapter_type"] == "lora"
    assert flavor_conf["adapter_weights"] == "adapter_weights"


def test_save_model_normalizes_single_file_input(adapter_file, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_file),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    weights_dir = model_path / "adapter_weights"
    assert weights_dir.exists()
    assert (weights_dir / "pytorch_lora_weights.safetensors").exists()


def test_save_model_normalizes_single_file_directory(adapter_dir, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    weights_dir = model_path / "adapter_weights"
    assert weights_dir.exists()
    assert (weights_dir / "pytorch_lora_weights.safetensors").exists()


def test_save_model_writes_environment_files(adapter_dir, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    assert (model_path / "conda.yaml").exists()
    assert (model_path / "requirements.txt").exists()
    assert (model_path / "python_env.yaml").exists()


def test_save_model_default_signature(adapter_dir, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    with open(model_path / MLMODEL_FILE_NAME) as f:
        mlmodel = yaml.safe_load(f)

    assert "signature" in mlmodel
    sig = mlmodel["signature"]
    inputs = json.loads(sig["inputs"])
    assert inputs[0]["name"] == "prompt"
    assert inputs[0]["type"] == "string"


def test_save_model_custom_signature(adapter_dir, model_path):
    from mlflow.types import DataType, Schema
    from mlflow.types.schema import ColSpec

    custom_sig = mlflow.models.ModelSignature(
        inputs=Schema([ColSpec(type=DataType.string, name="text")]),
        outputs=Schema([ColSpec(type=DataType.binary, name="img")]),
    )

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
        signature=custom_sig,
    )

    with open(model_path / MLMODEL_FILE_NAME) as f:
        mlmodel = yaml.safe_load(f)

    inputs = json.loads(mlmodel["signature"]["inputs"])
    assert inputs[0]["name"] == "text"


def test_save_model_invalid_adapter_type(adapter_dir, model_path):
    with pytest.raises(mlflow.exceptions.MlflowException, match="Unsupported adapter type"):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
            adapter_type="invalid",
        )


def test_save_model_nonexistent_path(model_path):
    with pytest.raises(mlflow.exceptions.MlflowException, match="does not exist"):
        mlflow.diffusers.save_model(
            adapter_path="/nonexistent/path",
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )


def test_save_model_with_metadata(adapter_dir, model_path):
    test_metadata = {"training_dataset": "my-dataset", "lora_rank": 16}

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
        metadata=test_metadata,
    )

    with open(model_path / MLMODEL_FILE_NAME) as f:
        mlmodel = yaml.safe_load(f)

    assert mlmodel["metadata"] == test_metadata


def test_log_model(adapter_dir):
    with mlflow.start_run() as run:
        model_info = mlflow.diffusers.log_model(
            adapter_path=str(adapter_dir),
            base_model=BASE_MODEL_ID,
            name="test_adapter",
        )

    assert model_info is not None

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "test_adapter")]
    assert any("adapter_weights" in a for a in artifacts)
    assert any(MLMODEL_FILE_NAME in a for a in artifacts)


def test_save_load_model_direct_roundtrip(adapter_dir, model_path):
    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    loaded = mlflow.diffusers.load_model(str(model_path))

    assert isinstance(loaded, DiffusersAdapterModel)
    assert loaded.base_model == BASE_MODEL_ID
    assert loaded.adapter_type == "lora"
    assert Path(loaded.adapter_path).exists()


def test_base_model_revision_roundtrip(adapter_dir, model_path):

    fake_revision = "abc123def456"
    with patch("mlflow.diffusers._resolve_base_model_revision", return_value=fake_revision):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )

    # Verify revision stored in flavor config
    mlmodel_path = model_path / MLMODEL_FILE_NAME
    with open(mlmodel_path) as f:
        mlmodel = yaml.safe_load(f)
    assert mlmodel["flavors"][FLAVOR_NAME]["base_model_revision"] == fake_revision

    # Verify revision survives load_model roundtrip
    loaded = mlflow.diffusers.load_model(str(model_path))
    assert loaded.base_model_revision == fake_revision


def test_load_model_via_tracking_roundtrip(adapter_dir):
    with mlflow.start_run() as run:
        mlflow.diffusers.log_model(
            adapter_path=str(adapter_dir),
            base_model=BASE_MODEL_ID,
            name="test_adapter",
        )

    model_uri = f"runs:/{run.info.run_id}/test_adapter"
    loaded = mlflow.diffusers.load_model(model_uri)

    assert isinstance(loaded, DiffusersAdapterModel)
    assert loaded.base_model == BASE_MODEL_ID
    assert loaded.adapter_type == "lora"
    assert Path(loaded.adapter_path).exists()
    assert (Path(loaded.adapter_path) / "pytorch_lora_weights.safetensors").exists()


def test_load_pyfunc_returns_wrapper(adapter_dir):
    from mlflow.diffusers.wrapper import _DiffusersAdapterWrapper

    with mlflow.start_run() as run:
        mlflow.diffusers.log_model(
            adapter_path=str(adapter_dir),
            base_model=BASE_MODEL_ID,
            name="test_adapter",
        )

    model_uri = f"runs:/{run.info.run_id}/test_adapter"
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)

    assert hasattr(loaded_pyfunc, "predict")
    wrapper = loaded_pyfunc._model_impl
    assert isinstance(wrapper, _DiffusersAdapterWrapper)
    assert wrapper._flavor_conf["base_model"] == BASE_MODEL_ID


def test_load_pipeline_base_model_override(adapter_dir, model_path):

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    loaded = mlflow.diffusers.load_model(str(model_path))
    override = "other-org/other-model"

    with patch("diffusers.DiffusionPipeline.from_pretrained") as mock_fp:
        mock_pipe = MagicMock()
        mock_fp.return_value = mock_pipe
        loaded.load_pipeline(base_model=override)
        mock_fp.assert_called_once()
        assert mock_fp.call_args[0][0] == override


def test_load_pipeline_wraps_oserror(adapter_dir, model_path):

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    loaded = mlflow.diffusers.load_model(str(model_path))

    with patch(
        "diffusers.DiffusionPipeline.from_pretrained",
        side_effect=OSError("is not a local folder"),
    ):
        with pytest.raises(MlflowException, match="Failed to load base model"):
            loaded.load_pipeline()


def test_wrapper_model_config_base_model_override():
    from mlflow.diffusers.wrapper import _DiffusersAdapterWrapper

    w = _DiffusersAdapterWrapper(
        adapter_path="/fake",
        flavor_conf={"base_model": "original/model", "adapter_type": "lora"},
        model_config={"base_model": "override/model"},
    )

    with patch("diffusers.DiffusionPipeline.from_pretrained") as mock_fp:
        mock_pipe = MagicMock()
        mock_fp.return_value = mock_pipe
        w._load_pipeline()
        mock_fp.assert_called_once()
        assert mock_fp.call_args[0][0] == "override/model"


@pytest.mark.parametrize("package", ["diffusers", "transformers", "torch", "peft", "safetensors"])
def test_default_pip_requirements_contains_core(package):
    reqs = mlflow.diffusers.get_default_pip_requirements()
    req_names = [r.split("==")[0] for r in reqs]
    assert package in req_names


@pytest.mark.parametrize("package", ["accelerate"])
def test_default_pip_requirements_contains_optional_if_installed(package):
    reqs = mlflow.diffusers.get_default_pip_requirements()
    req_names = [r.split("==")[0] for r in reqs]
    if importlib.util.find_spec(package):
        assert package in req_names
    else:
        assert package not in req_names


def test_default_conda_env():
    env = mlflow.diffusers.get_default_conda_env()
    assert "dependencies" in env


# -- predict() tests (mock-based, no GPU) --


class _FakeImage:
    def save(self, buf, format=None):
        buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)


class _FakePipelineOutput:
    def __init__(self, n=1):
        self.images = [_FakeImage() for _ in range(n)]


@pytest.fixture
def wrapper():
    from mlflow.diffusers.wrapper import _DiffusersAdapterWrapper

    w = _DiffusersAdapterWrapper(
        adapter_path="/fake",
        flavor_conf={"base_model": "test/model", "adapter_type": "lora"},
    )
    # Inject a mock pipeline so _load_pipeline is never called
    w._pipeline = lambda prompt, **kwargs: _FakePipelineOutput(len(prompt))
    return w


def test_get_raw_model(wrapper):
    model = wrapper.get_raw_model()
    assert model is not None
    assert callable(model)


def test_predict_dataframe(wrapper):
    import pandas as pd

    result = wrapper.predict(pd.DataFrame({"prompt": ["a cat", "a dog"]}))
    assert len(result) == 2
    assert all(isinstance(r, bytes) for r in result)


def test_predict_string(wrapper):
    result = wrapper.predict("a cat")
    assert len(result) == 1
    assert result[0][:4] == b"\x89PNG"


def test_predict_dict(wrapper):
    result = wrapper.predict({"prompt": "a cat"})
    assert len(result) == 1


def test_predict_dict_list(wrapper):
    result = wrapper.predict({"prompt": ["a cat", "a dog"]})
    assert len(result) == 2


def test_predict_list(wrapper):
    result = wrapper.predict(["a cat", "a dog"])
    assert len(result) == 2


def test_predict_with_params(wrapper):

    mock_pipeline = Mock(return_value=_FakePipelineOutput(1))
    wrapper._pipeline = mock_pipeline

    result = wrapper.predict(
        "a cat",
        params={"num_inference_steps": 10, "height": 256, "negative_prompt": "blurry"},
    )
    assert len(result) == 1

    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 10
    assert call_kwargs["height"] == 256
    assert call_kwargs["negative_prompt"] == "blurry"
    assert call_kwargs["prompt"] == ["a cat"]


def test_predict_dataframe_missing_prompt(wrapper):
    import pandas as pd

    with pytest.raises(MlflowException, match="prompt"):
        wrapper.predict(pd.DataFrame({"text": ["a cat"], "other": ["extra"]}))


def test_predict_dict_missing_prompt(wrapper):
    with pytest.raises(MlflowException, match="prompt"):
        wrapper.predict({"text": "a cat"})


def test_predict_empty_prompts(wrapper):
    with pytest.raises(MlflowException, match="No prompts"):
        wrapper.predict([])


def test_predict_dict_invalid_prompt_type(wrapper):
    with pytest.raises(MlflowException, match="must be a string or list of strings"):
        wrapper.predict({"prompt": 42})


def test_predict_unsupported_type(wrapper):
    with pytest.raises(MlflowException, match="Unsupported input type"):
        wrapper.predict(12345)


def test_predict_no_images(wrapper):
    class _NoImagesOutput:
        images = None

    wrapper._pipeline = lambda prompt, **kwargs: _NoImagesOutput()

    with pytest.raises(MlflowException, match="Pipeline returned no images"):
        wrapper.predict("a cat")


def test_predict_empty_images_list(wrapper):
    class _EmptyImagesOutput:
        images = []

    wrapper._pipeline = lambda prompt, **kwargs: _EmptyImagesOutput()

    with pytest.raises(MlflowException, match="Pipeline returned no images"):
        wrapper.predict("a cat")


def test_predict_none_prompt_rejected(wrapper):
    with pytest.raises(MlflowException, match="must be strings, not None"):
        wrapper.predict([None])


def test_predict_none_in_dataframe_rejected(wrapper):
    import pandas as pd

    with pytest.raises(MlflowException, match="must be strings, not None"):
        wrapper.predict(pd.DataFrame({"prompt": [None]}))


# -- _resolve_base_model_revision tests --


def test_resolve_revision_absolute_path_returns_none():
    from mlflow.diffusers import _resolve_base_model_revision

    assert _resolve_base_model_revision("/absolute/path/to/model") is None


def test_resolve_revision_dot_relative_path_returns_none():
    from mlflow.diffusers import _resolve_base_model_revision

    assert _resolve_base_model_revision("./local/model") is None
    assert _resolve_base_model_revision("../parent/model") is None


def test_resolve_revision_hf_hub_success():

    from mlflow.diffusers import _resolve_base_model_revision

    fake_sha = "abc123"
    with patch(
        "mlflow.utils.huggingface_utils.get_latest_commit_for_repo",
        return_value=fake_sha,
    ):
        result = _resolve_base_model_revision("org/model-name")
    assert result == fake_sha


def test_resolve_revision_hf_hub_failure_returns_none():

    from mlflow.diffusers import _resolve_base_model_revision

    with patch(
        "mlflow.utils.huggingface_utils.get_latest_commit_for_repo",
        side_effect=Exception("network error"),
    ):
        result = _resolve_base_model_revision("org/model-name")
    assert result is None


def test_resolve_revision_cwd_collision_still_resolves(tmp_path, monkeypatch):

    from mlflow.diffusers import _resolve_base_model_revision

    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    fake_sha = "deadbeef"
    with patch(
        "mlflow.utils.huggingface_utils.get_latest_commit_for_repo",
        return_value=fake_sha,
    ):
        result = _resolve_base_model_revision("org/model")
    assert result == fake_sha


def test_save_model_multi_file_directory_preserved(tmp_path, model_path):
    adapter_dir = tmp_path / "multi_adapter"
    adapter_dir.mkdir()
    tensors = {"w": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / "pytorch_lora_weights.safetensors"))
    # Companion file (e.g., text encoder LoRA or adapter_config from PEFT)
    (adapter_dir / "adapter_config.json").write_text('{"type": "lora"}')

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    weights_dir = model_path / "adapter_weights"
    assert (weights_dir / "pytorch_lora_weights.safetensors").exists()
    assert (weights_dir / "adapter_config.json").exists()


def test_save_model_records_weight_name_for_nonstandard_files(tmp_path, model_path):
    adapter_dir = tmp_path / "multi_nonstandard"
    adapter_dir.mkdir()
    tensors = {"w": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / "alpha_weights.safetensors"))
    save_file(tensors, str(adapter_dir / "beta_weights.safetensors"))

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    mlmodel = yaml.safe_load((model_path / MLMODEL_FILE_NAME).read_text())
    flavor_conf = mlmodel["flavors"]["diffusers"]
    assert flavor_conf["weight_name"] == "alpha_weights.safetensors"

    loaded = mlflow.diffusers.load_model(str(model_path))
    assert loaded.weight_name == "alpha_weights.safetensors"


def test_save_model_records_weight_name_for_peft_adapter(tmp_path, model_path):
    adapter_dir = tmp_path / "peft_adapter"
    adapter_dir.mkdir()
    tensors = {"w": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text('{"type": "lora"}')

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    mlmodel = yaml.safe_load((model_path / MLMODEL_FILE_NAME).read_text())
    flavor_conf = mlmodel["flavors"]["diffusers"]
    assert flavor_conf["weight_name"] == "adapter_model.safetensors"

    loaded = mlflow.diffusers.load_model(str(model_path))
    assert loaded.weight_name == "adapter_model.safetensors"


def test_save_model_rejects_non_safetensors_file(tmp_path, model_path):
    bad_file = tmp_path / "model.bin"
    bad_file.write_bytes(b"\x00" * 100)

    with pytest.raises(MlflowException, match=".safetensors"):
        mlflow.diffusers.save_model(
            adapter_path=str(bad_file),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )


def test_save_model_rejects_invalid_safetensors_content(tmp_path, model_path):
    fake_file = tmp_path / "bad.safetensors"
    fake_file.write_bytes(b"this is not safetensors format")

    with pytest.raises(MlflowException, match="not a valid safetensors"):
        mlflow.diffusers.save_model(
            adapter_path=str(fake_file),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )


@pytest.mark.parametrize(
    ("base_model", "match"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        (None, "must be a"),
        (123, "must be a"),
    ],
)
def test_save_model_rejects_invalid_base_model(adapter_dir, model_path, base_model, match):
    with pytest.raises(MlflowException, match=match):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model=base_model,
        )


@pytest.mark.parametrize("adapter_type", [None, 123, []])
def test_save_model_rejects_non_string_adapter_type(adapter_dir, model_path, adapter_type):
    with pytest.raises(MlflowException, match="adapter_type must be a string"):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
            adapter_type=adapter_type,
        )


# -- _detect_device tests --


def test_detect_device_explicit():
    from mlflow.diffusers import _detect_device

    assert _detect_device("cpu") == "cpu"
    assert _detect_device("cuda:1") == "cuda:1"


def test_detect_device_env_var(monkeypatch):
    from mlflow.diffusers import _detect_device

    monkeypatch.setenv("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
    assert _detect_device() == "cpu"


def test_predict_single_column_dataframe(wrapper):
    import pandas as pd

    result = wrapper.predict(pd.DataFrame(["a cat", "a dog"]))
    assert len(result) == 2


def test_save_model_rejects_empty_directory(tmp_path, model_path):
    empty_dir = tmp_path / "empty_adapter"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("no weights here")

    with pytest.raises(MlflowException, match="no .safetensors"):
        mlflow.diffusers.save_model(
            adapter_path=str(empty_dir),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )


def test_save_model_ignores_hidden_files(tmp_path, model_path):
    adapter_dir = tmp_path / "adapter_with_ds_store"
    adapter_dir.mkdir()
    tensors = {"w": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / "my_lora.safetensors"))
    (adapter_dir / ".DS_Store").write_bytes(b"\x00" * 10)

    mlflow.diffusers.save_model(
        adapter_path=str(adapter_dir),
        path=str(model_path),
        base_model=BASE_MODEL_ID,
    )

    weights_dir = model_path / "adapter_weights"
    assert (weights_dir / "pytorch_lora_weights.safetensors").exists()


def test_save_model_validates_all_safetensors_in_multi_file_dir(tmp_path, model_path):
    adapter_dir = tmp_path / "corrupt_multi"
    adapter_dir.mkdir()
    tensors = {"w": np.random.randn(4, 4).astype(np.float32)}
    save_file(tensors, str(adapter_dir / "pytorch_lora_weights.safetensors"))
    (adapter_dir / "corrupt.safetensors").write_bytes(b"not valid safetensors")

    with pytest.raises(MlflowException, match="not a valid safetensors"):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model=BASE_MODEL_ID,
        )

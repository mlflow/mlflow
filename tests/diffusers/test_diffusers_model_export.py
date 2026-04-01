import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

import mlflow
import mlflow.diffusers
from mlflow.diffusers import FLAVOR_NAME, DiffusersAdapterModel
from mlflow.models.model import MLMODEL_FILE_NAME


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


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


class TestSaveModel:
    def test_save_model_from_directory(self, adapter_dir, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
        )

        mlmodel_path = model_path / MLMODEL_FILE_NAME
        assert mlmodel_path.exists()

        with open(mlmodel_path) as f:
            mlmodel = yaml.safe_load(f)

        assert FLAVOR_NAME in mlmodel["flavors"]
        assert "python_function" in mlmodel["flavors"]

        flavor_conf = mlmodel["flavors"][FLAVOR_NAME]
        assert flavor_conf["base_model_id"] == BASE_MODEL_ID
        assert flavor_conf["adapter_type"] == "lora"
        assert flavor_conf["adapter_weights"] == "adapter_weights"

    def test_save_model_from_file(self, adapter_file, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_file),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
        )

        weights_dir = model_path / "adapter_weights"
        assert weights_dir.exists()
        assert (weights_dir / "adapter.safetensors").exists()

    def test_save_model_copies_adapter_weights(self, adapter_dir, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
        )

        weights_dir = model_path / "adapter_weights"
        assert weights_dir.exists()
        assert (weights_dir / "adapter.safetensors").exists()

    def test_save_model_writes_adapter_config(self, adapter_dir, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
            adapter_type="lora",
        )

        config_path = model_path / "adapter_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert config["base_model_id"] == BASE_MODEL_ID
        assert config["adapter_type"] == "lora"

    def test_save_model_writes_environment_files(self, adapter_dir, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
        )

        assert (model_path / "conda.yaml").exists()
        assert (model_path / "requirements.txt").exists()
        assert (model_path / "python_env.yaml").exists()

    def test_save_model_default_signature(self, adapter_dir, model_path):
        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
        )

        with open(model_path / MLMODEL_FILE_NAME) as f:
            mlmodel = yaml.safe_load(f)

        assert "signature" in mlmodel
        sig = mlmodel["signature"]
        inputs = json.loads(sig["inputs"])
        assert inputs[0]["name"] == "prompt"
        assert inputs[0]["type"] == "string"

    def test_save_model_custom_signature(self, adapter_dir, model_path):
        from mlflow.types import DataType, Schema
        from mlflow.types.schema import ColSpec

        custom_sig = mlflow.models.ModelSignature(
            inputs=Schema([ColSpec(type=DataType.string, name="text")]),
            outputs=Schema([ColSpec(type=DataType.binary, name="img")]),
        )

        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
            signature=custom_sig,
        )

        with open(model_path / MLMODEL_FILE_NAME) as f:
            mlmodel = yaml.safe_load(f)

        inputs = json.loads(mlmodel["signature"]["inputs"])
        assert inputs[0]["name"] == "text"

    def test_save_model_invalid_adapter_type(self, adapter_dir, model_path):
        with pytest.raises(mlflow.exceptions.MlflowException, match="Unsupported adapter type"):
            mlflow.diffusers.save_model(
                adapter_path=str(adapter_dir),
                path=str(model_path),
                base_model_id=BASE_MODEL_ID,
                adapter_type="invalid",
            )

    def test_save_model_nonexistent_path(self, model_path):
        with pytest.raises(mlflow.exceptions.MlflowException, match="does not exist"):
            mlflow.diffusers.save_model(
                adapter_path="/nonexistent/path",
                path=str(model_path),
                base_model_id=BASE_MODEL_ID,
            )

    def test_save_model_with_metadata(self, adapter_dir, model_path):
        test_metadata = {"training_dataset": "my-dataset", "lora_rank": 16}

        mlflow.diffusers.save_model(
            adapter_path=str(adapter_dir),
            path=str(model_path),
            base_model_id=BASE_MODEL_ID,
            metadata=test_metadata,
        )

        with open(model_path / MLMODEL_FILE_NAME) as f:
            mlmodel = yaml.safe_load(f)

        assert mlmodel["metadata"] == test_metadata


class TestLogModel:
    def test_log_model(self, adapter_dir):
        with mlflow.start_run() as run:
            model_info = mlflow.diffusers.log_model(
                adapter_path=str(adapter_dir),
                base_model_id=BASE_MODEL_ID,
                name="test_adapter",
            )

        assert model_info is not None

        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id, "test_adapter")]
        assert any("adapter_weights" in a for a in artifacts)
        assert any(MLMODEL_FILE_NAME in a for a in artifacts)

    def test_log_model_with_registered_name(self, adapter_dir):
        with mlflow.start_run():
            model_info = mlflow.diffusers.log_model(
                adapter_path=str(adapter_dir),
                base_model_id=BASE_MODEL_ID,
                name="test_adapter",
                registered_model_name="my-lora-model",
            )

        assert model_info is not None


class TestLoadModel:
    def test_load_model_roundtrip(self, adapter_dir):
        with mlflow.start_run() as run:
            mlflow.diffusers.log_model(
                adapter_path=str(adapter_dir),
                base_model_id=BASE_MODEL_ID,
                name="test_adapter",
            )

        model_uri = f"runs:/{run.info.run_id}/test_adapter"
        loaded = mlflow.diffusers.load_model(model_uri)

        assert isinstance(loaded, DiffusersAdapterModel)
        assert loaded.base_model_id == BASE_MODEL_ID
        assert loaded.adapter_type == "lora"
        assert Path(loaded.adapter_path).exists()
        assert (Path(loaded.adapter_path) / "adapter.safetensors").exists()


class TestLoadPyfunc:
    def test_load_pyfunc_returns_wrapper(self, adapter_dir):
        from mlflow.diffusers.wrapper import _DiffusersAdapterWrapper

        with mlflow.start_run() as run:
            mlflow.diffusers.log_model(
                adapter_path=str(adapter_dir),
                base_model_id=BASE_MODEL_ID,
                name="test_adapter",
            )

        model_uri = f"runs:/{run.info.run_id}/test_adapter"
        loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)

        assert hasattr(loaded_pyfunc, "predict")
        wrapper = loaded_pyfunc._model_impl
        assert isinstance(wrapper, _DiffusersAdapterWrapper)
        assert wrapper._flavor_conf["base_model_id"] == BASE_MODEL_ID


class TestPipRequirements:
    @pytest.mark.parametrize("package", ["diffusers", "transformers", "torch"])
    def test_default_pip_requirements_contains_core(self, package):
        reqs = mlflow.diffusers.get_default_pip_requirements()
        req_names = [r.split("==")[0] for r in reqs]
        assert package in req_names

    @pytest.mark.parametrize("package", ["accelerate", "safetensors", "peft"])
    def test_default_pip_requirements_contains_optional_if_installed(self, package):
        import importlib.util

        reqs = mlflow.diffusers.get_default_pip_requirements()
        req_names = [r.split("==")[0] for r in reqs]
        if importlib.util.find_spec(package):
            assert package in req_names
        else:
            assert package not in req_names

    def test_default_conda_env(self):
        env = mlflow.diffusers.get_default_conda_env()
        assert "dependencies" in env

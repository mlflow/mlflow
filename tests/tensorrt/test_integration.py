import builtins
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.onnx

import mlflow
from mlflow.models import infer_signature


def _require_trt_and_cuda_or_skip():
    """Check for TensorRT and CUDA availability, skip tests if unavailable.

    Returns:
        module: The tensorrt module if available.

    Raises:
        pytest.skip: If tensorrt cannot be imported or CUDA is not available.
    """
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        pytest.skip("tensorrt is required for these tests", allow_module_level=True)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TensorRT integration tests", allow_module_level=True)
    return trt


@pytest.fixture(scope="module")
def torch_linear_model():
    """Create a simple PyTorch linear model on CUDA for testing.

    Returns:
        torch.nn.Module: A linear model (4 inputs, 3 outputs) in eval mode on CUDA.
    """
    model = torch.nn.Linear(4, 3).cuda().eval()
    with torch.no_grad():
        # initialize deterministically
        torch.manual_seed(0)
        _ = model(torch.randn(2, 4, device="cuda"))
    return model


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory, torch_linear_model):
    """Export the PyTorch model to ONNX format.

    Args:
        tmp_path_factory: Pytest factory for creating temporary directories.
        torch_linear_model: The PyTorch model to export.

    Returns:
        str: Path to the exported ONNX model file.
    """
    path = Path(tmp_path_factory.mktemp("trt_int")) / "model.onnx"
    x = torch.randn(2, 4, device="cuda")
    torch.onnx.export(
        torch_linear_model,
        x,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    return str(path)


@pytest.fixture(scope="module")
def tensorrt_engine(onnx_model_path):
    """Build a TensorRT engine from the ONNX model.

    Args:
        onnx_model_path: Path to the ONNX model file.

    Returns:
        tensorrt.ICudaEngine: The compiled TensorRT engine.

    Raises:
        pytest.skip: If ONNX parsing or engine building fails.
    """
    trt = _require_trt_and_cuda_or_skip()

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, "rb") as f:
        parsed = parser.parse(f.read())
        if not parsed:
            # Collect errors for visibility
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            pytest.skip("Failed to parse ONNX for TensorRT: " + "; ".join(errs))

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    # Set profile for input name "input" with shape ranges
    inp = network.get_input(0)
    profile.set_shape(inp.name, (1, 4), (2, 4), (8, 4))
    config.add_optimization_profile(profile)

    runtime = trt.Runtime(logger)

    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        pytest.skip("Failed to build TensorRT engine")
    return engine


def test_save_load_and_pyfunc_predict(tmp_path, tensorrt_engine):
    """Test save/load flow and pyfunc prediction with a real TensorRT engine.

    Args:
        tmp_path: Pytest temporary directory fixture.
        tensorrt_engine: A compiled TensorRT engine fixture.
    """
    # Save engine via mlflow and load pyfunc
    model_dir = tmp_path / "trt_model"
    mlflow.tensorrt.save_model(trt_engine=tensorrt_engine, path=str(model_dir))

    # Load native
    loaded_engine = mlflow.tensorrt.load_model(str(model_dir))
    assert loaded_engine is not None

    # Load pyfunc and run prediction on CUDA
    wrapper = mlflow.pyfunc.load_model(str(model_dir))

    # Build an input dict based on engine bindings
    import tensorrt as trt  # type: ignore

    names = []
    for i in range(int(loaded_engine.num_io_tensors)):
        name = loaded_engine.get_tensor_name(i)
        if loaded_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            names.append(name)

    assert len(names) == 1
    batch = 2
    x = torch.randn(batch, 4, device="cuda", dtype=torch.float32)
    outs = wrapper.predict({names[0]: x})
    # Expect at least one output with batch dimension matching
    assert isinstance(outs, dict)
    assert any(tuple(v.shape)[0] == batch for v in outs.values())


def test_get_default_conda_env_and_pip_requirements():
    """Test that default environment helpers return valid configurations."""
    env = mlflow.tensorrt.get_default_conda_env()
    assert isinstance(env, dict)
    assert "dependencies" in env
    reqs = mlflow.tensorrt.get_default_pip_requirements()
    assert isinstance(reqs, list)
    assert len(reqs) > 0


def test_save_model_with_signature_and_metadata(tmp_path, tensorrt_engine):
    """Test saving a model with signature, input example, and metadata."""
    from mlflow.models import ModelSignature
    from mlflow.models.signature import Schema

    metadata = {"test_key": "test_value"}
    input_example = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    signature = infer_signature(input_example)

    model_dir = tmp_path / "trt_model_sig"
    mlflow.tensorrt.save_model(
        trt_engine=tensorrt_engine,
        path=str(model_dir),
        signature=signature,
        input_example=input_example,
        metadata=metadata,
    )

    from mlflow.models import Model

    mlmodel = Model.load(str(model_dir))
    assert mlmodel.signature == signature
    assert mlmodel.metadata == metadata
    assert mlmodel.saved_input_example_info is not None


def test_save_model_with_custom_pip_requirements(tmp_path, tensorrt_engine):
    """Test saving a model with custom pip requirements."""
    model_dir = tmp_path / "trt_model_pip"
    mlflow.tensorrt.save_model(
        trt_engine=tensorrt_engine,
        path=str(model_dir),
        pip_requirements=["torch==2.0.0", "numpy>=1.20"],
    )

    reqs_file = model_dir / "requirements.txt"
    assert reqs_file.exists()
    reqs_content = reqs_file.read_text()
    assert "torch==2.0.0" in reqs_content
    assert "numpy>=1.20" in reqs_content


def test_save_model_with_conda_env(tmp_path, tensorrt_engine):
    """Test saving a model with a custom conda environment."""
    import yaml

    conda_env = {
        "name": "test_env",
        "channels": ["defaults"],
        "dependencies": ["python=3.9", {"pip": ["tensorrt"]}],
    }

    model_dir = tmp_path / "trt_model_conda"
    mlflow.tensorrt.save_model(
        trt_engine=tensorrt_engine, path=str(model_dir), conda_env=conda_env
    )

    conda_file = model_dir / "conda.yaml"
    assert conda_file.exists()
    with open(conda_file) as f:
        saved_env = yaml.safe_load(f)
    assert saved_env["name"] == "test_env"


def test_log_model_and_retrieve(tensorrt_engine):
    """Test logging a model and retrieving it from MLflow tracking."""
    with mlflow.start_run():
        model_info = mlflow.tensorrt.log_model(trt_engine=tensorrt_engine, name="trt_model")

    assert model_info is not None
    assert model_info.model_uri is not None

    # Load the logged model
    loaded_engine = mlflow.tensorrt.load_model(model_info.model_uri)
    assert loaded_engine is not None


def test_pyfunc_get_raw_model(tmp_path, tensorrt_engine):
    """Test that pyfunc wrapper exposes the underlying TensorRT engine via get_raw_model()."""
    model_dir = tmp_path / "trt_model_raw"
    mlflow.tensorrt.save_model(trt_engine=tensorrt_engine, path=str(model_dir))

    wrapper = mlflow.pyfunc.load_model(str(model_dir))
    raw_engine = wrapper._model_impl.get_raw_model()
    assert raw_engine is not None


def test_pyfunc_predict_invalid_input_type_raises(tmp_path, tensorrt_engine):
    """Test that pyfunc predict raises TypeError for non-dict input."""
    model_dir = tmp_path / "trt_model_invalid"
    mlflow.tensorrt.save_model(trt_engine=tensorrt_engine, path=str(model_dir))

    wrapper = mlflow.pyfunc.load_model(str(model_dir))

    with pytest.raises(TypeError, match="TensorRT pyfunc expects a dict"):
        wrapper.predict([[1, 2, 3]])


def test_save_model_without_pynvml(tmp_path, tensorrt_engine, monkeypatch):
    """Test save_model when pynvml is not available (ImportError branch)."""
    # Block pynvml import to hit the except ImportError branch (lines 96-98, 100)
    import sys

    original_import = builtins.__import__

    def _block_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pynvml)

    model_dir = tmp_path / "trt_model_no_pynvml"
    mlflow.tensorrt.save_model(trt_engine=tensorrt_engine, path=str(model_dir))
    assert (model_dir / "MLmodel").exists()


def test_save_model_without_torch(tmp_path, tensorrt_engine, monkeypatch):
    """Test save_model when torch is not available (ImportError branch)."""
    # Block torch import to hit the except ImportError branch (lines 110-111)
    import sys

    original_import = builtins.__import__

    def _block_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torch)

    model_dir = tmp_path / "trt_model_no_torch"
    mlflow.tensorrt.save_model(trt_engine=tensorrt_engine, path=str(model_dir))
    assert (model_dir / "MLmodel").exists()


def test_save_model_with_invalid_engine_type_raises(tmp_path):
    """Test that save_model raises TypeError for non-ICudaEngine objects."""
    # Cover line 116 - TypeError for non-ICudaEngine
    with pytest.raises(TypeError, match="should be a tensorrt.ICudaEngine"):
        mlflow.tensorrt.save_model(trt_engine="not an engine", path=str(tmp_path / "invalid"))


def test_save_model_with_extra_pip_requirements_and_constraints(tmp_path, tensorrt_engine):
    """Test saving a model with extra pip requirements containing requirements file references."""
    # Use constraint syntax to hit line 190 (pip_constraints write)
    model_dir = tmp_path / "trt_model_constraints"
    mlflow.tensorrt.save_model(
        trt_engine=tensorrt_engine,
        path=str(model_dir),
        extra_pip_requirements=["-r requirements/test-requirements.txt", "somepackage"],
    )

    with open(model_dir / "requirements.txt") as f:
        content = f.read()
        assert "pytest" in content  # from test-requirements.txt

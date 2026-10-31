import pytest

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema


@pytest.fixture(autouse=True)
def mock_log_model_with_pip_requirements():
    from unittest import mock

    original_log_model = mlflow.pyfunc.log_model

    def patched_log_model(*args, **kwargs):
        kwargs.setdefault("pip_requirements", [])
        return original_log_model(*args, **kwargs)

    with mock.patch("mlflow.pyfunc.log_model", patched_log_model):
        yield


def test_log_model_preserves_user_params_schema_when_type_hint_inference_returns_none():
    """
    Regression test for https://github.com/mlflow/mlflow/issues/14908
    """

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(
            self,
            context,
            model_input: list[str],
            params: dict[str, str] | None = None,
        ) -> list[str]:
            return model_input

    signature = ModelSignature(
        inputs=Schema([ColSpec(type="string")]),
        outputs=Schema([ColSpec(type="string")]),
        params=ParamSchema([ParamSpec(name="k", dtype="long", default=1)]),
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=MyModel(),
            signature=signature,
        )

    assert model_info.signature.params is not None
    assert model_info.signature.params == signature.params

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = loaded_model.predict(["test"], params={"k": 1})
    assert result == ["test"]

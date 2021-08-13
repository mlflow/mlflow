import mlflow
from mlflow.utils.requirements_utils import (
    _module_to_packages,
    _capture_imported_modules,
    _infer_requirements,
)


class MlflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        import mlflow  # pylint: disable=unused-import

    def predict(self, _, __):
        pass


def test_infer_requirements_excludes_mlflow(tmpdir):
    mlflow_package = _module_to_packages("mlflow")[0]
    assert mlflow_package in ["mlflow", "mlflow-skinny"]

    model_path = tmpdir.join("model").strpath
    mlflow.pyfunc.save_model(model_path, python_model=MlflowModel())
    # Ensure mlflow is imported while loading the model
    assert "mlflow" in _capture_imported_modules(model_path, mlflow.pyfunc.FLAVOR_NAME)
    # Ensure inferred requirements don't contain `mlflow_package`
    inferred_reqs = _infer_requirements(model_path, mlflow.pyfunc.FLAVOR_NAME)
    assert all(not r.startswith(mlflow_package) for r in inferred_reqs)

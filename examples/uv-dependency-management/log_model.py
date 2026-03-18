"""
Example: Using uv for dependency management with MLflow models.

This script demonstrates three ways to use uv lockfile-based dependencies
when logging MLflow models:

1. Auto-detection: MLflow detects uv.lock + pyproject.toml in the current
   working directory and uses ``uv export`` to capture pinned dependencies.

2. Explicit path (uv_project_path): Point to a uv project directory when
   logging from a different working directory or in a monorepo layout.

3. Dependency groups and extras (uv_groups, uv_extras): Include additional
   dependency groups or optional extras defined in pyproject.toml.

Prerequisites:
    - uv >= 0.5.0 installed (``pip install uv`` or https://docs.astral.sh/uv/)
    - Run from this directory so auto-detection finds uv.lock and pyproject.toml

Usage:
    cd examples/uv-dependency-management
    uv run python log_model.py
"""

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import mlflow


def read_requirements(run_id, artifact_name="model"):
    """Read the model's requirements.txt from local run artifacts."""
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, f"{artifact_name}/requirements.txt")
    with open(local_path) as f:
        return [line.strip() for line in f if line.strip()]


def check_uv_artifacts(run_id, artifact_name="model"):
    """Check if uv project files were saved as model artifacts."""
    client = mlflow.tracking.MlflowClient()
    model_dir = client.download_artifacts(run_id, artifact_name)
    model_path = Path(model_dir)
    return {
        "uv.lock": (model_path / "uv.lock").exists(),
        "pyproject.toml": (model_path / "pyproject.toml").exists(),
    }


def train_model():
    """Train a simple RandomForest on the Iris dataset."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, X_test, accuracy


class SklearnWrapper(mlflow.pyfunc.PythonModel):
    """Wrap a scikit-learn model as a PythonModel for pyfunc logging."""

    def __init__(self, sklearn_model):
        self._model = sklearn_model

    def predict(self, context, model_input, params=None):
        return self._model.predict(model_input)


def example_auto_detection(model, input_example):
    """
    Example 1: Auto-detection.

    When run from a directory containing uv.lock and pyproject.toml,
    MLflow automatically uses uv export to capture pinned dependencies.
    No extra parameters needed.
    """
    print("=" * 60)
    print("Example 1: Auto-detection")
    print("=" * 60)

    with mlflow.start_run(run_name="uv-auto-detection") as run:
        model_info = mlflow.pyfunc.log_model(
            python_model=SklearnWrapper(model),
            name="model",
            input_example=input_example,
        )

        run_id = run.info.run_id
        reqs = read_requirements(run_id)
        print(f"Logged model: {model_info.model_uri}")
        print(f"Requirements ({len(reqs)} packages):")
        for req in reqs[:10]:
            print(f"  {req}")
        if len(reqs) > 10:
            print(f"  ... and {len(reqs) - 10} more")

        # Verify uv artifacts were saved
        uv_files = check_uv_artifacts(run_id)
        print(f"uv.lock saved as artifact: {uv_files['uv.lock']}")
        print(f"pyproject.toml saved as artifact: {uv_files['pyproject.toml']}")
        print()

    return model_info


def example_explicit_path(model, input_example):
    """
    Example 2: Explicit uv_project_path.

    Use uv_project_path to point to a uv project when logging from
    a different working directory. Useful in monorepos.
    """
    print("=" * 60)
    print("Example 2: Explicit uv_project_path")
    print("=" * 60)

    project_dir = Path(__file__).parent.resolve()

    with mlflow.start_run(run_name="uv-explicit-path") as run:
        model_info = mlflow.pyfunc.log_model(
            python_model=SklearnWrapper(model),
            name="model",
            input_example=input_example,
            uv_project_path=project_dir,
        )

        run_id = run.info.run_id
        reqs = read_requirements(run_id)
        print(f"Logged model: {model_info.model_uri}")
        print(f"uv_project_path: {project_dir}")
        print(f"Requirements ({len(reqs)} packages):")
        for req in reqs[:10]:
            print(f"  {req}")
        if len(reqs) > 10:
            print(f"  ... and {len(reqs) - 10} more")
        print()

    return model_info


def example_groups_and_extras(model, input_example):
    """
    Example 3: Dependency groups and extras.

    Include the 'ml' dependency group (xgboost) and 'serving' optional
    extra (flask) in the exported requirements.
    """
    print("=" * 60)
    print("Example 3: uv_groups and uv_extras")
    print("=" * 60)

    project_dir = Path(__file__).parent.resolve()

    with mlflow.start_run(run_name="uv-groups-and-extras") as run:
        model_info = mlflow.pyfunc.log_model(
            python_model=SklearnWrapper(model),
            name="model",
            input_example=input_example,
            uv_project_path=project_dir,
            uv_groups=["ml"],
            uv_extras=["serving"],
        )

        run_id = run.info.run_id
        reqs = read_requirements(run_id)
        print(f"Logged model: {model_info.model_uri}")
        print("uv_groups: ['ml']  (adds xgboost)")
        print("uv_extras: ['serving']  (adds flask)")
        print(f"Requirements ({len(reqs)} packages):")

        # Check that group and extra deps were included
        has_xgboost = any("xgboost" in r for r in reqs)
        has_flask = any("flask" in r.lower() for r in reqs)
        print(f"  xgboost included (from 'ml' group): {has_xgboost}")
        print(f"  flask included (from 'serving' extra): {has_flask}")
        print()

        for req in sorted(reqs):
            print(f"  {req}")
        print()

    return model_info


def main():
    print("MLflow uv Dependency Management Example")
    print()

    # Train a model
    model, X_test, accuracy = train_model()
    input_example = X_test[:2]
    print(f"Trained RandomForestClassifier (accuracy: {accuracy:.2f})")
    print()

    # Set up a local tracking URI for the example.
    # Use an absolute path so it works regardless of working directory.
    example_dir = Path(__file__).parent.resolve()
    db_path = example_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("uv-dependency-management")

    # Run all three examples
    example_auto_detection(model, input_example)
    example_explicit_path(model, input_example)
    example_groups_and_extras(model, input_example)

    print("All examples completed successfully.")


if __name__ == "__main__":
    main()

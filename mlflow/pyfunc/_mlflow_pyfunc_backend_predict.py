"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse

from mlflow.pyfunc.scoring_server import _predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--input-path", required=False)
    parser.add_argument("--output-path", required=False)
    parser.add_argument("--content-type", required=True)
    return parser.parse_args()


# Guidance for fixing missing module error
_MISSING_MODULE_HELP_MSG = (
    "Exception occurred while running inference: {e}"
    "\n\n"
    "\033[93m[Hint] It appears that your MLflow Model doesn't contain the required "
    "dependency '{missing_module}' to run model inference. When logging a model, MLflow "
    "detects dependencies based on the model flavor, but it is possible that some "
    "dependencies are not captured. In this case, you can manually add dependencies "
    "using the `extra_pip_requirements` parameter of `mlflow.pyfunc.log_model`.\033[0m"
    """

\033[1mSample code:\033[0m
    ----
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=your_model,
        extra_pip_requirements=["{missing_module}==x.y.z"]
    )
    ----

    For mode guidance on fixing missing dependencies, please refer to the MLflow docs:
    https://www.mlflow.org/docs/latest/deployment/index.html#how-to-fix-dependency-errors-when-serving-my-model
"""
)


def main():
    args = parse_args()

    try:
        _predict(
            model_uri=args.model_uri,
            input_path=args.input_path if args.input_path else None,
            output_path=args.output_path if args.output_path else None,
            content_type=args.content_type,
        )
    except ModuleNotFoundError as e:
        message = _MISSING_MODULE_HELP_MSG.format(e=str(e), missing_module=e.name)
        raise RuntimeError(message) from e


if __name__ == "__main__":
    main()

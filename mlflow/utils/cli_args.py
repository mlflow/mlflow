"""
Definitions of click options shared by several CLI commands.
"""
import click

MODEL_PATH = click.option("--model-path", "-m", default=None, metavar="PATH", required=True,
                          help="Path to the model. The path is relative to the run with the given "
                               "run-id or local filesystem path without run-id.")

MODEL_URI = click.option("--model-uri", "-m", default=None, metavar="URI", required=True,
                         help="URI to the model. This may be a local path, a 'runs:/' URI, or a"
                              " remote storage URI (e.g., an 's3://' URI). For more information"
                              " about supported remote URIs for model artifacts, see"
                              " https://mlflow.org/docs/latest/tracking.html"
                              "#supported-artifact-stores")

MLFLOW_HOME = click.option("--mlflow-home", default=None, metavar="PATH",
                           help="Path to local clone of MLflow project. Use for development only.")

RUN_ID = click.option("--run-id", "-r", default=None, required=False, metavar="ID",
                      help="ID of the MLflow run that generated the referenced content.")

NO_CONDA = click.option("--no-conda", is_flag=True,
                        help="If specified, will assume that MLModel/MLProject is running within "
                             "a Conda environmen with the necessary dependencies for "
                             "the current project instead of attempting to create a new "
                             "conda environment.")

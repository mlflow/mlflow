"""
Definitions of click options shared by several CLI commands.
"""
import click

MODEL_PATH = click.option("--model-path", "-m", default=None, metavar="PATH", required=True,
                          help="Path to the model. The path is relative to the run with the given "
                               "run-id or local filesystem path without run-id.")

MLFLOW_HOME = click.option("--mlflow-home", default=None, metavar="PATH",
                           help="Path to local clone of MLflow project. Use for development only.")

RUN_ID = click.option("--run-id", "-r", default=None, required=False, metavar="ID",
                      help="ID of the MLflow run that generated the referenced content.")

NO_CONDA = click.option("--no-conda", is_flag=True,
                        help="If specified, will assume that MLModel/MLProject is running within "
                             "a Conda environmen with the necessary dependencies for "
                             "the current project instead of attempting to create a new "
                             "conda environment.")

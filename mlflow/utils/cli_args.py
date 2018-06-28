"""
Definitions of Click.options shared by several CLI commands.
"""
import click

MODEL_PATH = click.option("--model-path", "-m", default=None, metavar="PATH", required=True,
                          help="Path to the model. The path is relative to the run with the given "
                               + "RUN_ID or local filesystem path without RUN_ID.")

MLFLOW_HOME = click.option("--mlflow-home", default=None, metavar="PATH",
                           help="Path to local clone of MLflow project. Use for development only.")

RUN_ID = click.option("--run-id", "-r", default=None, required=False, metavar="ID",
                      help="ID of the MLflow run that generated the referenced content.")

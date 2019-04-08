"""
File containing all of the run tags in the mlflow. namespace.

See the REST API documentation for information on the meaning of these tags.
"""

MLFLOW_RUN_NAME = "mlflow.runName"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_SOURCE_TYPE = "mlflow.source.type"
MLFLOW_SOURCE_NAME = "mlflow.source.name"
MLFLOW_GIT_COMMIT = "mlflow.source.git.commit"
MLFLOW_GIT_BRANCH = "mlflow.source.git.branch"
MLFLOW_GIT_REPO_URL = "mlflow.source.git.repoURL"
MLFLOW_PROJECT_ENV = "mlflow.project.env"
MLFLOW_PROJECT_ENTRY_POINT = "mlflow.project.entryPoint"
MLFLOW_DOCKER_IMAGE_NAME = "mlflow.docker.image.name"
MLFLOW_DOCKER_IMAGE_ID = "mlflow.docker.image.id"

MLFLOW_DATABRICKS_NOTEBOOK_ID = "mlflow.databricks.notebookID"
MLFLOW_DATABRICKS_NOTEBOOK_PATH = "mlflow.databricks.notebookPath"
MLFLOW_DATABRICKS_WEBAPP_URL = "mlflow.databricks.webappURL"
MLFLOW_DATABRICKS_RUN_URL = "mlflow.databricks.runURL"
MLFLOW_DATABRICKS_SHELL_JOB_ID = "mlflow.databricks.shellJobID"
MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = "mlflow.databricks.shellJobRunID"

# The following legacy tags are deprecated and will be removed by MLflow 1.0.
LEGACY_MLFLOW_GIT_BRANCH_NAME = "mlflow.gitBranchName"  # Replaced with mlflow.source.git.branch
LEGACY_MLFLOW_GIT_REPO_URL = "mlflow.gitRepoURL"  # Replaced with mlflow.source.git.repoURL

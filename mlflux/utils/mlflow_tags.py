"""
File containing all of the run tags in the mlflux. namespace.

See the System Tags section in the mlflux Tracking documentation for information on the
meaning of these tags.
"""

MLFLOW_RUN_NAME = "mlflux.runName"
MLFLOW_RUN_NOTE = "mlflux.note.content"
MLFLOW_PARENT_RUN_ID = "mlflux.parentRunId"
MLFLOW_USER = "mlflux.user"
MLFLOW_SOURCE_TYPE = "mlflux.source.type"
MLFLOW_SOURCE_NAME = "mlflux.source.name"
MLFLOW_GIT_COMMIT = "mlflux.source.git.commit"
MLFLOW_GIT_BRANCH = "mlflux.source.git.branch"
MLFLOW_GIT_REPO_URL = "mlflux.source.git.repoURL"
MLFLOW_LOGGED_MODELS = "mlflux.log-model.history"
MLFLOW_PROJECT_ENV = "mlflux.project.env"
MLFLOW_PROJECT_ENTRY_POINT = "mlflux.project.entryPoint"
MLFLOW_DOCKER_IMAGE_URI = "mlflux.docker.image.uri"
MLFLOW_DOCKER_IMAGE_ID = "mlflux.docker.image.id"
# Indicates that an mlflux run was created by an autologging integration
MLFLOW_AUTOLOGGING = "mlflux.autologging"

MLFLOW_DATABRICKS_NOTEBOOK_ID = "mlflux.databricks.notebookID"
MLFLOW_DATABRICKS_NOTEBOOK_PATH = "mlflux.databricks.notebookPath"
MLFLOW_DATABRICKS_WEBAPP_URL = "mlflux.databricks.webappURL"
MLFLOW_DATABRICKS_RUN_URL = "mlflux.databricks.runURL"
MLFLOW_DATABRICKS_CLUSTER_ID = "mlflux.databricks.cluster.id"
# The unique ID of a command execution in a Databricks notebook
MLFLOW_DATABRICKS_NOTEBOOK_COMMAND_ID = "mlflux.databricks.notebook.commandID"
# The SHELL_JOB_ID and SHELL_JOB_RUN_ID tags are used for tracking the
# Databricks Job ID and Databricks Job Run ID associated with an mlflux Project run
MLFLOW_DATABRICKS_SHELL_JOB_ID = "mlflux.databricks.shellJobID"
MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = "mlflux.databricks.shellJobRunID"
# The JOB_ID, JOB_RUN_ID, and JOB_TYPE tags are used for automatically recording Job information
# when mlflux Tracking APIs are used within a Databricks Job
MLFLOW_DATABRICKS_JOB_ID = "mlflux.databricks.jobID"
MLFLOW_DATABRICKS_JOB_RUN_ID = "mlflux.databricks.jobRunID"
MLFLOW_DATABRICKS_JOB_TYPE = "mlflux.databricks.jobType"


MLFLOW_PROJECT_BACKEND = "mlflux.project.backend"

# The following legacy tags are deprecated and will be removed by mlflux 1.0.
LEGACY_MLFLOW_GIT_BRANCH_NAME = "mlflux.gitBranchName"  # Replaced with mlflux.source.git.branch
LEGACY_MLFLOW_GIT_REPO_URL = "mlflux.gitRepoURL"  # Replaced with mlflux.source.git.repoURL

"""
File containing all of the run tags in the mlflow. namespace.

See the System Tags section in the MLflow Tracking documentation for information on the
meaning of these tags.
"""

MLFLOW_EXPERIMENT_SOURCE_ID = "mlflow.experiment.sourceId"
MLFLOW_EXPERIMENT_SOURCE_TYPE = "mlflow.experiment.sourceType"
MLFLOW_RUN_NAME = "mlflow.runName"
MLFLOW_RUN_NOTE = "mlflow.note.content"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_ARTIFACT_LOCATION = "mlflow.artifactLocation"
MLFLOW_USER = "mlflow.user"
MLFLOW_SOURCE_TYPE = "mlflow.source.type"
MLFLOW_SOURCE_NAME = "mlflow.source.name"
MLFLOW_GIT_COMMIT = "mlflow.source.git.commit"
MLFLOW_GIT_BRANCH = "mlflow.source.git.branch"
MLFLOW_GIT_DIRTY = "mlflow.source.git.dirty"
MLFLOW_GIT_REPO_URL = "mlflow.source.git.repoURL"
MLFLOW_GIT_DIFF = "mlflow.source.git.diff"
MLFLOW_LOGGED_MODELS = "mlflow.log-model.history"
MLFLOW_MODEL_IS_EXTERNAL = "mlflow.model.isExternal"
MLFLOW_MODEL_VERSIONS = "mlflow.modelVersions"
MLFLOW_PROJECT_ENV = "mlflow.project.env"
MLFLOW_PROJECT_ENTRY_POINT = "mlflow.project.entryPoint"
MLFLOW_DOCKER_IMAGE_URI = "mlflow.docker.image.uri"
MLFLOW_DOCKER_IMAGE_ID = "mlflow.docker.image.id"
# Indicates that an MLflow run was created by an autologging integration
MLFLOW_AUTOLOGGING = "mlflow.autologging"
# Indicates the artifacts type and path that are logged
MLFLOW_LOGGED_ARTIFACTS = "mlflow.loggedArtifacts"
MLFLOW_LOGGED_IMAGES = "mlflow.loggedImages"
MLFLOW_RUN_SOURCE_TYPE = "mlflow.runSourceType"

# Indicates that an MLflow run was created by an evaluation
MLFLOW_RUN_IS_EVALUATION = "mlflow.run.isEval"

MLFLOW_DATABRICKS_NOTEBOOK_ID = "mlflow.databricks.notebookID"
MLFLOW_DATABRICKS_NOTEBOOK_PATH = "mlflow.databricks.notebookPath"
MLFLOW_DATABRICKS_WEBAPP_URL = "mlflow.databricks.webappURL"
MLFLOW_DATABRICKS_RUN_URL = "mlflow.databricks.runURL"
MLFLOW_DATABRICKS_CLUSTER_ID = "mlflow.databricks.cluster.id"
MLFLOW_DATABRICKS_WORKSPACE_URL = "mlflow.databricks.workspaceURL"
MLFLOW_DATABRICKS_WORKSPACE_ID = "mlflow.databricks.workspaceID"
# The unique ID of a command execution in a Databricks notebook
MLFLOW_DATABRICKS_NOTEBOOK_COMMAND_ID = "mlflow.databricks.notebook.commandID"
# The SHELL_JOB_ID and SHELL_JOB_RUN_ID tags are used for tracking the
# Databricks Job ID and Databricks Job Run ID associated with an MLflow Project run
MLFLOW_DATABRICKS_SHELL_JOB_ID = "mlflow.databricks.shellJobID"
MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = "mlflow.databricks.shellJobRunID"
# The JOB_ID, JOB_RUN_ID, and JOB_TYPE tags are used for automatically recording Job information
# when MLflow Tracking APIs are used within a Databricks Job
MLFLOW_DATABRICKS_JOB_ID = "mlflow.databricks.jobID"
MLFLOW_DATABRICKS_JOB_RUN_ID = "mlflow.databricks.jobRunID"
# Here MLFLOW_DATABRICKS_JOB_TYPE means the job task type and MLFLOW_DATABRICKS_JOB_TYPE_INFO
# implies the job type which could be normal, ephemeral, etc.
MLFLOW_DATABRICKS_JOB_TYPE = "mlflow.databricks.jobType"
MLFLOW_DATABRICKS_JOB_TYPE_INFO = "mlflow.databricks.jobTypeInfo"
# For MLflow Repo Lineage tracking
MLFLOW_DATABRICKS_GIT_REPO_URL = "mlflow.databricks.gitRepoUrl"
MLFLOW_DATABRICKS_GIT_REPO_COMMIT = "mlflow.databricks.gitRepoCommit"
MLFLOW_DATABRICKS_GIT_REPO_PROVIDER = "mlflow.databricks.gitRepoProvider"
MLFLOW_DATABRICKS_GIT_REPO_RELATIVE_PATH = "mlflow.databricks.gitRepoRelativePath"
MLFLOW_DATABRICKS_GIT_REPO_REFERENCE = "mlflow.databricks.gitRepoReference"
MLFLOW_DATABRICKS_GIT_REPO_REFERENCE_TYPE = "mlflow.databricks.gitRepoReferenceType"
MLFLOW_DATABRICKS_GIT_REPO_STATUS = "mlflow.databricks.gitRepoStatus"

# Databricks model serving endpoint information
MLFLOW_DATABRICKS_MODEL_SERVING_ENDPOINT_NAME = "mlflow.databricks.modelServingEndpointName"

# For Serverless GPU Compute (SGC) run resumption
# Experiment tag prefix that maps SGC job run IDs to MLflow run IDs for automatic resumption
# Format: mlflow.databricks.sgc.resumeRun.jobRunId.{job_run_id} -> {mlflow_run_id}
MLFLOW_DATABRICKS_SGC_RESUME_RUN_JOB_RUN_ID_PREFIX = "mlflow.databricks.sgc.resumeRun.jobRunId"

# For MLflow Dataset tracking
MLFLOW_DATASET_CONTEXT = "mlflow.data.context"

MLFLOW_PROJECT_BACKEND = "mlflow.project.backend"

MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME = "mlflow.experiment.primaryMetric.name"
MLFLOW_EXPERIMENT_PRIMARY_METRIC_GREATER_IS_BETTER = (
    "mlflow.experiment.primaryMetric.greaterIsBetter"
)

# For automatic model checkpointing
LATEST_CHECKPOINT_ARTIFACT_TAG_KEY = "mlflow.latest_checkpoint_artifact"


# A set of tags that cannot be updated by the user
IMMUTABLE_TAGS = {MLFLOW_USER, MLFLOW_ARTIFACT_LOCATION}

# The list of tags generated from resolve_tags() that are required for tracing UI
TRACE_RESOLVE_TAGS_ALLOWLIST = (
    MLFLOW_DATABRICKS_NOTEBOOK_COMMAND_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_DIRTY,
)


def _get_run_name_from_tags(tags):
    for tag in tags:
        if tag.key == MLFLOW_RUN_NAME:
            return tag.value

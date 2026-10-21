# CLAUDE.md

## Keep the job containers docs up to date

[`docs/docs/self-hosting/job-containers.mdx`](../../../docs/docs/self-hosting/job-containers.mdx) tells users how the code in this directory behaves when it runs code-based scorers in job containers. When a change affects any of the following, update that page in the same PR:

- The settings it documents (`MLFLOW_SERVER_JOB_EXECUTION_ENGINE`, `MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND`, `MLFLOW_JOB_IMAGE`, `MLFLOW_JOBS_DOCKER_HOST`, `MLFLOW_JOB_RESOURCE_LIMITS_<job name>`, `MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT`), including their defaults and limits.
- The default job image and how it is built.
- What a job container can do: its network, user, mounts and resource limits, and the labels on it.
- What the server and its job runner check when they start, and what happens when Docker or the job image is not available.
- Messages the page quotes, such as `The docker job executor runs custom scorers without network access`, `The docker job executor cannot reach the Docker daemon` and `Executor job runner exited unexpectedly`.

# CLAUDE.md

## Keep the job containers docs up to date

[`docs/docs/self-hosting/job-containers.mdx`](../../../../docs/docs/self-hosting/job-containers.mdx) tells users how automatic evaluation runs code-based scorers in job containers, which `container_scoring.py` implements. When a change affects any of the following, update that page in the same PR:

- How a job splits its traces or sessions into batches (`MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE`), and how the batches share the job's time limit.
- How a job retries setting up its containers when Docker is unreachable.
- Which failures give traces an error assessment, and whether those traces are scored again.

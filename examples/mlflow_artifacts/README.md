# MLflow Artifacts Example

This directory contains a set of files for demonstrating the MLflow Artifacts Service.

## What does the MLflow Artifacts Service do?

The MLflow Artifacts Service serves as a proxy between the client and artifact storage (e.g. S3)
and allows the client to upload, download, and list artifacts via REST API without configuring
a set of credentials required to access resources in the artifact storage (e.g. `AWS_ACCESS_KEY_ID`
and `AWS_SECRET_ACCESS_KEY` for S3).

## Quick start

First, launch the tracking server with the artifacts service via `mlflow server`:

```sh
# Launch a tracking server with the artifacts service
$ mlflow server \
    --backend-store-uri=mlruns \
    --artifacts-destination ./mlartifacts \
    --default-artifact-root http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/experiments \
    --gunicorn-opts "--log-level debug"
```

Notes:

- `--artifacts-destination` specifies the base artifact location from which to resolve artifact upload/download/list requests. In this examples, we're using a local directory `./mlartifacts`, but it can be changed to a s3 bucket or
- `--default-artifact-root` points to the `experiments` directory of the artifacts service. Therefore, the default artifact location of a newly-created experiment is set to `./mlartifacts/experiments/<experiment_id>`.
- `--gunicorn-opts "--log-level debug"` is specified to print out request logs but can be omitted if unnecessary.
- `--artifacts-only` disables all other endpoints for the tracking server apart from those involved in listing, uploading, and downloading artifacts. This makes the MLflow server a single-purpose proxy for artifact handling only.

Then, run `example.py` that performs upload, download, and list operations for artifacts:

```
$ MLFLOW_TRACKING_URI=http://localhost:5000 python example.py
```

After running the command above, the server should print out request logs for artifact operations:

```diff
...
[2021-11-05 19:13:34 +0900] [92800] [DEBUG] POST /api/2.0/mlflow/runs/create
[2021-11-05 19:13:34 +0900] [92800] [DEBUG] GET /api/2.0/mlflow/runs/get
[2021-11-05 19:13:34 +0900] [92802] [DEBUG] PUT /api/2.0/mlflow-artifacts/artifacts/0/a1b2c3d4/artifacts/a.txt
[2021-11-05 19:13:34 +0900] [92802] [DEBUG] PUT /api/2.0/mlflow-artifacts/artifacts/0/a1b2c3d4/artifacts/dir/b.txt
[2021-11-05 19:13:34 +0900] [92802] [DEBUG] POST /api/2.0/mlflow/runs/update
[2021-11-05 19:13:34 +0900] [92802] [DEBUG] GET /api/2.0/mlflow-artifacts/artifacts
...
```

The contents of the `mlartifacts` directory should look like this:

```sh
$ tree mlartifacts
mlartifacts
└── experiments
    └── 0  # experiment ID
        └── a1b2c3d4  # run ID
            └── artifacts
                ├── a.txt
                └── dir
                    └── b.txt

5 directories, 2 files
```

To delete the logged artifacts, run the following command:

```bash
mlflow gc --backend-store-uri=mlruns --run-ids
```

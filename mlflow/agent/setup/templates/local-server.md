### Start a local MLflow tracking server

No tracking URI was provided. An available port was already picked: use
`{tracking_uri}` as the tracking URI for the rest of this task.

1. Add a short note to the README with the corresponding `mlflow server`
   command so the user can restart the server later. Match the host and port
   to `{tracking_uri}` and prefix with the project's package manager where
   appropriate (e.g. `uv run mlflow server ...`).
2. During verification (step 4), start the server in the background with logs
   redirected to a temp file. Kill it once verification passes so no orphan
   process is left behind.

### Start a local MLflow tracking server

No tracking URI was provided. An available port was already picked: use
`{tracking_uri}` as the tracking URI for the rest of this task.

During verification (step 4), start the server in the background with logs
redirected to a temp file:

```
mlflow server --host 127.0.0.1 --port {port} > /tmp/mlflow-server.log 2>&1 &
```

Leave it running afterward so the user can open the trace URL in the MLflow
UI. Report the PID and the log file path in the final summary so the user
knows how to stop it.

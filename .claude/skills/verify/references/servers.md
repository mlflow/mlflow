# Servers

Run one checkout's servers at a time. Set the paths for the PR run as shown below.
For the base run, set `checkout="$tmp_dir/checkouts/base"` and `run_dir="$tmp_dir/work/base"`.

```bash
checkout="$pr_checkout"
run_dir="$tmp_dir/work/pr"
```

## Backend server

REST API and UI checks need a backend. Use one worker, so a request always reaches the process
you are watching:

```bash
mkdir -p "$run_dir"
cd "$checkout"
uv run mlflow server --dev --workers 1 --host localhost --port 5000 \
  --backend-store-uri "sqlite:///$run_dir/mlflow.db" \
  --artifacts-destination "$run_dir/artifacts" \
  > "$run_dir/backend.log" 2>&1 &
echo $! > "$run_dir/backend.pid"
```

Poll until it is ready, checking that the process is alive on every attempt:

```bash
kill -0 "$(cat "$run_dir/backend.pid")" &&
  curl -fsS --max-time 2 http://localhost:5000/health >/dev/null
```

Without the `kill -0`, a server that crashed at startup makes you wait out the whole timeout. If
the process has exited, read its log, fix the cause, and retry right away.

## Frontend dev server

UI checks also need the frontend. Use `localhost` for readiness checks and browser URLs:

```bash
cd "$checkout/mlflow/server/js"
yarn install --immutable
PORT=3000 BROWSER=none CI=false yarn start > "$run_dir/frontend.log" 2>&1 &
echo $! > "$run_dir/frontend.pid"
```

```bash
kill -0 "$(cat "$run_dir/frontend.pid")" &&
  curl -fsS --max-time 2 http://localhost:3000/ >/dev/null
```

For a backend on another port, set `MLFLOW_PROXY=http://localhost:<port>/`. Unless
`MLFLOW_DEV_PROXY_MODE` is set, that switches the frontend to HTTPS: use `curl -k` with
`https://localhost:<frontend_port>/`, and pass `--ignore-https-errors` on the browser's first
`open`.

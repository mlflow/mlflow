# Servers

When comparing against the base, run the base and PR servers one after the other, and stop one
before starting the next.

## Backend server

REST API and UI checks need a backend. Use one worker, so a request always reaches the process
you are watching:

```bash
cd "$pr_checkout"
uv run mlflow server --dev --workers 1 --host localhost --port 5000 \
  --backend-store-uri "sqlite:///$tmp_dir/work/mlflow.db" \
  --artifacts-destination "$tmp_dir/work/artifacts" \
  > "$tmp_dir/work/backend.log" 2>&1 &
echo $! > "$tmp_dir/work/backend.pid"
```

Poll until it is ready, checking that the process is alive on every attempt:

```bash
kill -0 "$(cat "$tmp_dir/work/backend.pid")" &&
  curl -fsS --max-time 2 http://localhost:5000/health >/dev/null
```

Without the `kill -0`, a server that crashed at startup makes you wait out the whole timeout. If
the process has exited, read its log, fix the cause, and retry right away.

## Frontend dev server

UI checks also need the frontend. Use `localhost` for readiness checks and browser URLs:

```bash
cd "$pr_checkout/mlflow/server/js"
yarn install --immutable
PORT=3000 BROWSER=none CI=false yarn start > "$tmp_dir/work/frontend.log" 2>&1 &
echo $! > "$tmp_dir/work/frontend.pid"
```

```bash
kill -0 "$(cat "$tmp_dir/work/frontend.pid")" &&
  curl -fsS --max-time 2 http://localhost:3000/ >/dev/null
```

For a backend on another port, set `MLFLOW_PROXY=http://localhost:<port>/`. Unless
`MLFLOW_DEV_PROXY_MODE` is set, that switches the frontend to HTTPS: use `curl -k` with
`https://localhost:<frontend_port>/`, and pass `--ignore-https-errors` on the browser's first
`open`.

# Servers

Run one checkout's servers at a time. The commands below use `$pr_checkout` for the source
and `$tmp_dir/work/pr` for its data, logs, and PID files. For the base run, use
`$tmp_dir/checkouts/base` and `$tmp_dir/work/base` in their place throughout.

## Backend server

REST API and UI checks need a backend. Use one worker, so a request always reaches the process
you are watching:

```bash
mkdir -p "$tmp_dir/work/pr"
cd "$pr_checkout" || exit 1
uv run mlflow server --dev --workers 1 --host localhost --port 5000 \
  --backend-store-uri "sqlite:///$tmp_dir/work/pr/mlflow.db" \
  --artifacts-destination "$tmp_dir/work/pr/artifacts" \
  > "$tmp_dir/work/pr/backend.log" 2>&1 &
echo $! > "$tmp_dir/work/pr/backend.pid"
```

Poll until it is ready, checking that the process is alive on every attempt:

```bash
kill -0 "$(cat "$tmp_dir/work/pr/backend.pid")" &&
  curl -fsS --max-time 2 http://localhost:5000/health >/dev/null
```

Without the `kill -0`, a server that crashed at startup makes you wait out the whole timeout. If
the process has exited, read its log, fix the cause, and retry right away.

## Frontend dev server

UI checks also need the frontend. Use `localhost` for readiness checks and browser URLs:

```bash
mkdir -p "$tmp_dir/work/pr"
cd "$pr_checkout/mlflow/server/js" || exit 1
yarn install --immutable
PORT=3000 BROWSER=none CI=false yarn start > "$tmp_dir/work/pr/frontend.log" 2>&1 &
echo $! > "$tmp_dir/work/pr/frontend.pid"
```

```bash
kill -0 "$(cat "$tmp_dir/work/pr/frontend.pid")" &&
  curl -fsS --max-time 2 http://localhost:3000/ >/dev/null
```

For a backend on another port, set `MLFLOW_PROXY=http://localhost:<port>/`. Unless
`MLFLOW_DEV_PROXY_MODE` is set, that switches the frontend to HTTPS: use `curl -k` with
`https://localhost:<frontend_port>/`, and pass `--ignore-https-errors` on the browser's first
`open`.

## Switch checkouts

Stop the outgoing servers using their recorded PIDs. A worker can outlive `yarn start`
or the backend reloader. Check the default ports with `lsof -nP -iTCP:3000 -iTCP:5000 -sTCP:LISTEN`.
For any remaining listener, check its working directory with `lsof -a -p "$pid" -d cwd`
and stop it if it belongs to the outgoing checkout. Start the next checkout once both ports are free.

# Benchmarks

To compare the performance of the experimental Go implementation versus the current Python backend, we have set up a small K6 script. This script is meant to be run locally and gives us an initial performance impression.

## How to run

1. **Start the experimental tracking server with the Go flag:**

   ```sh
   mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres --experimental-go --experimental-go-opts LogLevel=debug,ShutdownTimeout=5s
   ```

   This command starts the Go server on port `5000` and the Python server on a random port. Check the logs to determine the Python server port:

   ```sh
   INFO[0000] Starting MLflow experimental Go server on http://127.0.0.1:5000
   DEBU[0000] Launching command: /usr/local/bin/python -m gunicorn -b 127.0.0.1:41603 -w 4 mlflow.server:app
   INFO[0000] [2024-06-10 12:34:07 +0000] [60150] [INFO] Starting gunicorn 22.0.0
   INFO[0000] [2024-06-10 12:34:07 +0000] [60150] [INFO] Listening at: http://127.0.0.1:41603 (60150)
   INFO[0000] [2024-06-10 12:34:07 +0000] [60150] [INFO] Using worker: sync
   INFO[0000] [2024-06-10 12:34:07 +0000] [60152] [INFO] Booting worker with pid: 60152
   INFO[0000] [2024-06-10 12:34:07 +0000] [60153] [INFO] Booting worker with pid: 60153
   INFO[0000] [2024-06-10 12:34:07 +0000] [60165] [INFO] Booting worker with pid: 60165
   INFO[0000] [2024-06-10 12:34:07 +0000] [60166] [INFO] Booting worker with pid: 60166
   ```

2. **Run the K6 script against the server:**

   For the Go server on port 5000:

   ```sh
   k6 run -e HOSTNAME='localhost:5000' ./dev/benchmarks/k6LogBatchPerfScript.js
   ```

   For the Python server on the random port (41603 in this example):

   ```sh
   k6 run -e HOSTNAME='localhost:41603' ./dev/benchmarks/k6LogBatchPerfScript.js
   ```

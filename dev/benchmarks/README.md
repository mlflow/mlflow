# Benchmarks

To compare the performance of the experimental Go implementation versus the current Python backend, we have set up a small K6 script. This script is meant to be run locally and gives us an initial performance impression.

## How to run

### Current tracking server

```sh
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres
```

Run test via:

```sh
k6 run -e MLFLOW_TRACKING_URI=http://localhost:5000 k6LogBatchPerfScript.js -u 20 -d 30s
```

### Experimental Go flag

```sh
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres --experimental-go --experimental-go-opts LogLevel=error
```

Run test via:

```sh
k6 run -e MLFLOW_TRACKING_URI=http://localhost:5000 k6LogBatchPerfScript.js -u 20 -d 30s
```

### Alternative configurations

To experiment with K6, the following flags can be tweaked:

- `-u`: number of virtual users (default 1).
- `-d`: test duration limit (eg. `30s`). This is interesting to compare how many requests were made.
- `-i`: script total iteration limit (among all VUs)

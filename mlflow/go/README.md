# Experimental Go

In order to increase the performance of the `mlflow server` command, we propose to rewrite the Python server implementation in Go.

## General setup

To ensure we stay compatible with the Python implementation, we aim to generate as much as possible based on the `.proto` files in `/mlflow/protos`.

By running `./dev/generate-protos.sh` Go code will be generated.
This incudes:

- Structs for each endpoint. (`/mlflow/go/pkg/protos`)
- Go interfaces for each service. (`/mlflow/go/pkg/contract/interfaces.g.go`)
- [fiber](https://gofiber.io/) routes for each point. (`/mlflow/go/pkg/contract/interfaces.g.go`)

If there is any change in the proto files, this should ripple into the Go code.

## Launching the Go server

To enable use the Go server, users can add the `--experimental-go` flag.

```bash
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres --experimental-go
```

This will launch the python process as usual. Within Python, a random port is chose to start the existing server.
And a Go child process is spawn. The Go server will use the user specified port (5000 by default).
Any incoming requests the Go server cannot process will be proxied to the existing Python server.

## Data access

Initially, we want to focus on supporting Postgres SQL. We chose [Gorm](https://gorm.io/) as ORM to interact with the database.

We do not generate any Go code based on the database schema. Gorm has generation capabilities but they didn't fit our needs. The plan would be to eventually assert the current code stil matches the database schema via an intergration test.

## Testing strategy

To ensure parity with the existing Python integration tests, we are investigating to re-use the Python integration tests.

## Supported endpoints

- [ ] Get /api/2.0/mlflow/experiments/get-by-name
- [x] Post /api/2.0/mlflow/experiments/create
- [ ] Post /api/2.0/mlflow/experiments/search
- [ ] Get /api/2.0/mlflow/experiments/search
- [x] Get /api/2.0/mlflow/experiments/get
- [ ] Post /api/2.0/mlflow/experiments/delete
- [ ] Post /api/2.0/mlflow/experiments/restore
- [ ] Post /api/2.0/mlflow/experiments/update
- [ ] Post /api/2.0/mlflow/runs/create
- [ ] Post /api/2.0/mlflow/runs/update
- [ ] Post /api/2.0/mlflow/runs/delete
- [ ] Post /api/2.0/mlflow/runs/restore
- [ ] Post /api/2.0/mlflow/runs/log-metric
- [ ] Post /api/2.0/mlflow/runs/log-parameter
- [ ] Post /api/2.0/mlflow/experiments/set-experiment-tag
- [ ] Post /api/2.0/mlflow/runs/set-tag
- [ ] Post /api/2.0/mlflow/runs/delete-tag
- [ ] Get /api/2.0/mlflow/runs/get
- [ ] Post /api/2.0/mlflow/runs/search
- [ ] Get /api/2.0/mlflow/artifacts/list
- [ ] Get /api/2.0/mlflow/metrics/get-history
- [ ] Get /api/2.0/mlflow/metrics/get-history-bulk-interval
- [ ] Post /api/2.0/mlflow/runs/log-batch
- [ ] Post /api/2.0/mlflow/runs/log-model
- [ ] Post /api/2.0/mlflow/runs/log-inputs
- [ ] Post /api/2.0/mlflow/registered-models/create
- [ ] Post /api/2.0/mlflow/registered-models/rename
- [ ] Patch /api/2.0/mlflow/registered-models/update
- [ ] Delete /api/2.0/mlflow/registered-models/delete
- [ ] Get /api/2.0/mlflow/registered-models/get
- [ ] Get /api/2.0/mlflow/registered-models/search
- [ ] Post /api/2.0/mlflow/registered-models/get-latest-versions
- [ ] Get /api/2.0/mlflow/registered-models/get-latest-versions
- [ ] Post /api/2.0/mlflow/model-versions/create
- [ ] Patch /api/2.0/mlflow/model-versions/update
- [ ] Post /api/2.0/mlflow/model-versions/transition-stage
- [ ] Delete /api/2.0/mlflow/model-versions/delete
- [ ] Get /api/2.0/mlflow/model-versions/get
- [ ] Get /api/2.0/mlflow/model-versions/search
- [ ] Get /api/2.0/mlflow/model-versions/get-download-uri
- [ ] Post /api/2.0/mlflow/registered-models/set-tag
- [ ] Post /api/2.0/mlflow/model-versions/set-tag
- [ ] Delete /api/2.0/mlflow/registered-models/delete-tag
- [ ] Delete /api/2.0/mlflow/model-versions/delete-tag
- [ ] Post /api/2.0/mlflow/registered-models/alias
- [ ] Delete /api/2.0/mlflow/registered-models/alias
- [ ] Get /api/2.0/mlflow/registered-models/alias
- [ ] Get /api/2.0/mlflow-artifacts/artifacts/:path
- [ ] Put /api/2.0/mlflow-artifacts/artifacts/:path
- [ ] Get /api/2.0/mlflow-artifacts/artifacts
- [ ] Delete /api/2.0/mlflow-artifacts/artifacts/:path
- [ ] Post /api/2.0/mlflow-artifacts/mpu/create/:path
- [ ] Post /api/2.0/mlflow-artifacts/mpu/complete/:path
- [ ] Post /api/2.0/mlflow-artifacts/mpu/abort/:path

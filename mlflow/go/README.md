# Experimental Go

In order to increase the performance of the `mlflow server` command, we propose to rewrite the Python server implementation in Go.

## General setup

To ensure we stay compatible with the Python implementation, we aim to generate as much as possible based on the `.proto` files in [/mlflow/protos](../protos/service.proto).

By running [dev/generate-protos.sh](../dev/generate-protos.sh) Go code will be generated.
This incudes:

- Structs for each endpoint. ([mlflow/go/pkg/protos](./pkg/protos/service.pb.go))
- Go interfaces for each service. ([mlflow/go/pkg/contract/interfaces.g.go](./pkg/contract/interface.g.go))
- [fiber](https://gofiber.io/) routes for each endpoint. ([mlflow/go/pkg/contract/interfaces.g.go](./pkg/contract/interface.g.go))

If there is any change in the proto files, this should ripple into the Go code.

## Launching the Go server

To enable use of the Go server, users can add the `--experimental-go` flag.

```bash
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres --experimental-go
```

This will launch the python process as usual. Within Python, a random port is chosen to start the existing server and a Go child process is spawned. The Go server will use the user specified port (5000 by default) and spawn the actual Python server as its own child process (`gunicorn` or `waitress`).
Any incoming requests the Go server cannot process will be proxied to the existing Python server.

Any Go-specific options can be passed with `--experimental-go-opts`, which takes a comma-separated list of key-value pairs.

```bash
mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/postgres --experimental-go --experimental-go-opts LogLevel=debug,ShutdownTimeout=5s
```

## Request validation

We use [Go validator](https://github.com/go-playground/validator) to validate all incoming request structs.
As the proto files don't specify any validation rules, we map them manually in [mlflow/go/tools/generate/validations.go](./tools/generate/validations.go).

Once the mapping has been done, validation will be invoked automatically in the generated fiber code.

When the need arises, we can write custom validation function in [mlflow/go/pkg/server/validation.go](./pkg/server/validation.go).

## Data access

Initially, we want to focus on supporting Postgres SQL. We chose [Gorm](https://gorm.io/) as ORM to interact with the database.

We do not generate any Go code based on the database schema. Gorm has generation capabilities but they didn't fit our needs. The plan would be to eventually assert the current code stil matches the database schema via an intergration test.

All the models use pointers for their fields. We do this for performance reasons and to distinguish between zero values and null values.

## Testing strategy

The Python integration tests have been adapted to also run against the Go implementation. Just run them as usual, e.g.

```bash
pytest tests/tracking/test_rest_tracking.py
```

To run only the tests targetting the Go implementation, you can use the `-k` flag:

```bash
pytest tests/tracking/test_rest_tracking.py -k '[go-'
```

If you'd like to run a specific test and see its output 'live', you can use the `-s` flag:

```bash
pytest -s "tests/tracking/test_rest_tracking.py::test_create_experiment_validation[go-postgresql]"
```

See the [pytest documentation](https://docs.pytest.org/en/8.2.x/how-to/usage.html#specifying-which-tests-to-run) for more details.

## Supported endpoints

The currently supported endpoints can be found in [mlflow/tools/generate/endspoints.go](./tools/generate/endspoints.go).

## Linters

We have enabled various linters from [golangci-lint](https://golangci-lint.run/), you can run these via:

```bash
pre-commit run golangci-lint --all-files
```

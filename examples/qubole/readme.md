# Running MLFlow in Qubole Mode

When run in `"qubole"` mode, a `ShellCommand` is launched on QDS from the MLFlow project. A `cluster-spec.json` must be passed as follows,

```json
{
    "aws": {
        "s3_experiment_bucket": "dev.canopydata.com",
        "s3_experiment_base_path": "ameya/mlfow-test"
    },
    "qubole": {
        "api_token": "xyz" ,
        "api_url": "https://api.qubole.com/api/",
        "version": "v1.2",
        "poll_interval": 5,
        "skip_ssl_cert_check": false,
        "cloud_name": "AWS"
    },
    "cluster": {
        "label": "mlflow-test"
    },
    "command": {
        "name": "mlflow-test",
        "tags": ["mlflow"],
        "notify": false
    }
}
```

A toy example can be launch using the following command,

```sh
mlflow run git@github.com:mlflow/mlflow-example.git -P alpha=0.5 -m qubole --cluster-spec example/qubole_run_remote/cluster_spec.json
```
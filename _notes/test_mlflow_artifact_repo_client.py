
def test_mlflow_artifact_repo_client():
    from mlflow import MlflowClient

    client = MlflowClient("http://localhost:5000")
    experiment_id = client.create_experiment("test")

    experiment = client.get_experiment(experiment_id)
    print("Artifact Location: {}".format(experiment.artifact_location))

    r = client.create_run(experiment_id=experiment_id)
    client.log_artifact(r.info.run_id, "./README.rst")


if __name__ == '__main__':
    test_mlflow_artifact_repo_client()

import mlflow


epochs = "30"
batch_size = "32"
if __name__ == '__main__':
    with mlflow.start_run() as run:
        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "model_type": "VGG16",
                "pretrained_weights": "imagenet",
                "epochs": epochs,
                "batch_size": batch_size},
            experiment_id=run.info.experiment_id,
            block=False,
            use_conda=False)
        p.wait()

        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "model_type": "VGG16",
                "pretrained_weights": "None",
                "epochs": epochs,
                "batch_size": batch_size},
            experiment_id=run.info.experiment_id,
            block=False,
            use_conda=False)
        p.wait()

        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "model_type": "Inception_V3",
                "pretrained_weights": "imagenet",
                "epochs": epochs,
                "batch_size": batch_size
            },
            experiment_id=run.info.experiment_id,
            block=False,
            use_conda=False)
        p.wait()

        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "model_type": "Inception_V3",
                "pretrained_weights": "None",
                "epochs": epochs,
                "batch_size": batch_size
            },
            experiment_id=run.info.experiment_id,
            block=False,
            use_conda=False)
        p.wait()


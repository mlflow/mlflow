import argparse
import mlflow
from ax.service.ax_client import AxClient
from iris import IrisClassification
from iris_data_module import IrisDataModule
import pytorch_lightning as pl


def train_evaluate(params, max_epochs):
    model = IrisClassification(**params)
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=max_epochs)
    mlflow.pytorch.autolog()
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    test_accuracy = trainer.callback_metrics.get("test_acc")
    return test_accuracy


def model_training_hyperparameter_tuning(max_epochs, total_trials, params):
    """
     This function takes input params max_epochs, total_trials, params
     and creates a nested run in Mlflow. The parameters, metrics, model and summary are dumped into their
     respective mlflow-run ids. The best parameters are dumped along with the baseline model.

    :param max_epochs: Max epochs used for training the model. Type:int
    :param total_trials: Number of ax-client experimental trials. Type:int
    :param params: Model parameters. Type:dict
    """
    with mlflow.start_run(run_name="Parent Run"):
        train_evaluate(params=params, max_epochs=max_epochs)

        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-3, 0.15], "log_scale": True},
                {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3]},
                {"name": "momentum", "type": "range", "bounds": [0.7, 1.0]},
            ],
            objective_name="test_accuracy",
        )

        for i in range(total_trials):
            with mlflow.start_run(nested=True, run_name="Trial " + str(i)) as child_run:
                parameters, trial_index = ax_client.get_next_trial()
                test_accuracy = train_evaluate(params=parameters, max_epochs=max_epochs)

                # completion of trial
                ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

        best_parameters, metrics = ax_client.get_best_parameters()
        for param_name, value in best_parameters.items():
            mlflow.log_param("optimum_" + param_name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=50,
        help="number of epochs",
    )

    parser.add_argument(
        "--total_trials",
        default=3,
        help="umber of trials to be run for the optimization experiment",
    )

    args = parser.parse_args()

    params = {"lr": 0.1, "momentum": 0.9, "weight_decay": 0}

    model_training_hyperparameter_tuning(
        max_epochs=int(args.max_epochs), total_trials=int(args.total_trials), params=params
    )

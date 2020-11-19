import argparse
import mlflow
from ax.service.ax_client import AxClient
import classifier


def model_training_hyperparameter_tuning(max_epochs, total_trials, params):
    """
     This function takes input params max_epcohs.epxeirment_name,total_trials,params and tracking_uri
     and creates a nested run in Mlflow.  The parameters,metrics,model and summary are dumped into their
     respective mlflow-run ids. The best parameters are dumped along with the basedline model.

    :param max_epochs:Max epochs used for training the model. Type:int
    :param experiment_name: Mlflow experiment name , in which the runs would be logged. Type:str
    :param total_trials: Number of ax-client experimental trials. Type:int
    :param params: Model parameters. Type:dict
    :param tracking_uri: Mlflow tracking_uri
    """
    mlflow.start_run(run_name="Parent Run")
    dm = classifier.DataModule()
    model = classifier.LeNet(kwargs=params)
    classifier.train_evaluate(dm=dm, model=model, max_epochs=max_epochs)

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-3, 0.15], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3]},
            {"name": "nesterov", "type": "choice", "values": [True, False]},
            {"name": "momentum", "type": "range", "bounds": [0.7, 1.0]},
        ],
        objective_name="test_accuracy",
    )

    total_trials = total_trials

    for i in range(total_trials):
        with mlflow.start_run(nested=True, run_name="Trial " + str(i)) as child_run:
            parameters, trial_index = ax_client.get_next_trial()
            dm = classifier.DataModule()
            model = classifier.LeNet(kwargs=parameters)
            # calling the model
            test_accuracy = classifier.train_evaluate(
                parameterization=None, dm=dm, model=model, max_epochs=max_epochs
            )

            # completion of trial
            ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

    best_parameters, metrics = ax_client.get_best_parameters()
    for param_name, value in best_parameters.items():
        mlflow.log_param("optimum " + param_name, value)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=2,
        help="Describes the number of times a neural network has to be trained",
    )

    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of trials to be run for the optimization experiment",
    )

    args = parser.parse_args()

    params = {"lr": 0.011, "momentum": 0.9, "weight_decay": 0, "nesterov": False}

    model_training_hyperparameter_tuning(
        max_epochs=int(args.max_epochs), total_trials=int(args.total_trials), params=params
    )

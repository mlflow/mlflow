# pylint: disable=W0221
# pylint: disable=W0201
# pylint: disable=W0223
# pylint: disable=arguments-differ
# pylint: disable=abstract-method

import copy
import tempfile

import mlflow.pytorch
import time
import shutil
from ax.service.ax_client import AxClient
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils import try_mlflow_log
from prettytable import PrettyTable
from torch.nn.utils import prune
import alexnet as classifier
import torch
import pytorch_lightning as pl
import argparse
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from pathlib import Path
import os


global pruning_amount


def load_model(artifact_uri):
    path = Path(_download_artifact_from_uri(artifact_uri))
    model_file_path = os.path.join(path, "model/data/model.pth")
    return torch.load(model_file_path)


def prune_and_save_model(model,amount):

    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            # m = prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            m = prune.l1_unstructured(module, name="weight", amount=amount)
            m = prune.remove(module, "weight")
            name = m.weight

        if isinstance(module, torch.nn.Linear):
            # prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            m = prune.l1_unstructured(module, name="weight", amount=amount)
            m = prune.remove(module, "weight")
            name = m.weight

    mlflow.pytorch.save_state_dict(model.state_dict(),".")
    m1 = torch.load("state_dict.pth")
    os.remove("state_dict.pth")
    return m1


def load_pruned_model(filename):
    checkpoint = torch.load(filename)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    return model


def count_model_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        param = parameter.nonzero(as_tuple=False).size(0)
        table.add_row([name, param])
        total_params += param
    return table, total_params


def iterative_prune(
    model, parametrization, trainer, dm, testloader, iteration_count
):
    global pruning_amount
    if iteration_count == 0:
        pruning_amount = parametrization.get("amount")
    else:
        pruning_amount += 0.15

    mlflow.log_metric("PRUNING PERCENTAGE", pruning_amount)
    pruned_model = prune_and_save_model(model,pruning_amount)
    model.load_state_dict(copy.deepcopy(pruned_model))
    summary, params = count_model_parameters(model)
    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "pruned_model_summary.txt")
        params = "Total Trainable Parameters :" + str(params)
        with open(summary_file, "w") as f:
            f.write(str(summary))
            f.write("\n")
            f.write(str(params))

        try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
    finally:
        shutil.rmtree(tempdir)

    mlflow.pytorch.autolog()
    trainer.fit(model, dm)

    trainer.test(datamodule=testloader)
    metrics = trainer.callback_metrics
    test_accuracy = metrics.get("avg_test_acc")
    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of AX trials to be run for the optimization experiment",
    )
    parser.add_argument(
        "--total_pruning_iterations",
        default=3,
        help="It indicated number of Iterative Pruning steps to be run on the base model",
    )
    parser.add_argument(
        "--mlflow_run_name",
        help="Name of MLFLOW experiment run with which iterations results have to be attached",
    )

    args = parser.parse_args()

    mlflow_experiment_name = args.mlflow_experiment_name
    run_name = args.mlflow_run_name

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    else:
        tracking_uri = "http://localhost:5000/"

    mlflow.tracking.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    identifier = client.get_experiment_by_name(mlflow_experiment_name)
    mlflow.set_experiment(mlflow_experiment_name)
    runs = client.search_runs(
        experiment_ids=identifier.experiment_id, run_view_type=ViewType.ACTIVE_ONLY
    )[0]
    runs_dict = dict(runs)
    run_id = runs_dict.get("info").run_id
    artifact_uri = runs_dict.get("info").artifact_uri

    model = load_model(artifact_uri)
    dm = classifier.DataModule()
    dm.setup("fit")
    testloader = dm.setup("test")

    mlflow.start_run(run_id=run_id, run_name=run_name)

    total_trials = int(args.total_trials)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "amount", "type": "range", "bounds": [0.05, 0.15], "value_type": "float"}
        ],
        objective_name="test_accuracy",
    )

    for k in range(total_trials):

        parameters, trial_index = ax_client.get_next_trial()
        x = parameters.get("amount")
        x = round(x, 3)
        for i in range(int(args.total_pruning_iterations)):
            with mlflow.start_run(nested=True, run_name="Iteration" + str(i)) as child_run:
                mlflow.set_tags({"AX_TRIAL": k})

                trainer = pl.Trainer(max_epochs=int(args.max_epochs))

                # calling the model
                test_accuracy = iterative_prune(
                    model, parameters, trainer, dm, testloader, i
                )

                # completion of trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

    mlflow.end_run()

import argparse
import copy
import os
import shutil
import tempfile
from pathlib import Path

import lightning as L
import torch
from ax.service.ax_client import AxClient
from prettytable import PrettyTable
from torch.nn.utils import prune

import mlflow.pytorch
from mnist import (
    MNISTDataModule,
    LightningMNISTClassifier,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


class IterativePrune:
    def __init__(self):
        self.parser_args = None
        self.ax_client = None
        self.base_model_path = "base_model"
        self.pruning_amount = None

    def run_mnist_model(self, base=False):
        if base:
            mlflow.start_run(run_name="BaseModel")
        mlflow.pytorch.autolog()
        dm = MNISTDataModule()
        dm.setup(stage="fit")

        model = LightningMNISTClassifier()
        trainer = L.Trainer(max_epochs=self.parser_args.max_epochs)
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)
        if os.path.exists(self.base_model_path):
            shutil.rmtree(self.base_model_path)
        mlflow.pytorch.save_model(trainer.lightning_module, self.base_model_path)
        return trainer

    def load_base_model(self):
        path = Path(_download_artifact_from_uri(self.base_model_path))
        model_file_path = os.path.join(path, "data/model.pth")
        return torch.load(model_file_path)

    def initialize_ax_client(self):
        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            parameters=[
                {"name": "amount", "type": "range", "bounds": [0.05, 0.15], "value_type": "float"}
            ],
            objective_name="test_accuracy",
        )

    @staticmethod
    def prune_and_save_model(model, amount):
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")

        mlflow.pytorch.save_state_dict(model.state_dict(), ".")
        model = torch.load("state_dict.pth")
        os.remove("state_dict.pth")
        return model

    @staticmethod
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

    @staticmethod
    def write_prune_summary(summary, params):
        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, "pruned_model_summary.txt")
            params = "Total Trainable Parameters :" + str(params)
            with open(summary_file, "w") as f:
                f.write(str(summary))
                f.write("\n")
                f.write(str(params))

            mlflow.log_artifact(local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def iterative_prune(self, model, parametrization):
        if not self.pruning_amount:
            self.pruning_amount = parametrization.get("amount")
        else:
            self.pruning_amount += 0.15

        mlflow.log_metric("PRUNING PERCENTAGE", self.pruning_amount)
        pruned_model = self.prune_and_save_model(model, self.pruning_amount)
        model.load_state_dict(copy.deepcopy(pruned_model))
        summary, params = self.count_model_parameters(model)
        self.write_prune_summary(summary, params)
        trainer = self.run_mnist_model()
        metrics = trainer.callback_metrics
        test_accuracy = metrics.get("avg_test_acc")
        return test_accuracy

    def initiate_pruning_process(self, model):
        total_trials = int(vars(self.parser_args)["total_trials"])

        trial_index = None
        for i in range(total_trials):
            parameters, trial_index = self.ax_client.get_next_trial()
            print("***************************************************************************")
            print("Running Trial {}".format(i + 1))
            print("***************************************************************************")
            with mlflow.start_run(nested=True, run_name="Iteration" + str(i)):
                mlflow.set_tags({"AX_TRIAL": i})

                # calling the model
                test_accuracy = self.iterative_prune(model, parameters)

                # completion of trial
        self.ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

        # Ending the Base run
        mlflow.end_run()

    def get_parser_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--max_epochs",
            default=3,
            type=int,
            help="Number of AX trials to be run for the optimization experiment",
        )

        parser.add_argument(
            "--total_trials",
            default=3,
            type=int,
            help="Number of AX trials to be run for the optimization experiment",
        )

        self.parser_args = parser.parse_args()


if __name__ == "__main__":
    # Initializing
    iterative_prune_obj = IterativePrune()

    # Deriving parser arguments
    iterative_prune_obj.get_parser_args()

    # Running the base model
    print("***************************************************************************")
    print("Running Base Model")
    print("***************************************************************************")
    iterative_prune_obj.run_mnist_model(base=True)

    # Iterative Pruning
    iterative_prune_obj.initialize_ax_client()
    base_model = iterative_prune_obj.load_base_model()
    iterative_prune_obj.initiate_pruning_process(base_model)

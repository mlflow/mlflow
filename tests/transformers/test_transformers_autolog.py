import random

import numpy as np
import optuna
import pytest
import sklearn
import sklearn.cluster
import torch
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, sample_dataset
from setfit import Trainer as SetFitTrainer
from setfit import TrainingArguments as SetFitTrainingArguments
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
)

import mlflow


# TODO: Remove this fixture once https://github.com/huggingface/transformers/pull/29096 is merged
@pytest.fixture(autouse=True)
def set_mlflow_tracking_uri_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())


@pytest.fixture
def iris_data():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


@pytest.fixture
def setfit_trainer():
    dataset = load_dataset("sst2")

    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
    eval_dataset = dataset["validation"]

    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    return SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        column_mapping={"sentence": "text", "label": "label"},
        args=SetFitTrainingArguments(
            loss=CosineSimilarityLoss,
            batch_size=16,
            num_iterations=5,
            num_epochs=1,
            report_to="none",
        ),
    )


@pytest.fixture
def transformers_trainer(tmp_path):
    random.seed(8675309)
    np.random.seed(8675309)
    torch.manual_seed(8675309)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    train_texts = ["I love this product!", "This is terrible."]
    train_labels = [1, 0]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)

    training_args = TrainingArguments(
        output_dir=str(tmp_path.joinpath("results")),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir=str(tmp_path.joinpath("logs")),
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )


@pytest.fixture
def transformers_hyperparameter_trainer(tmp_path):
    random.seed(555)
    np.random.seed(555)
    torch.manual_seed(555)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    train_texts = ["I love this product!", "This is terrible."]
    train_labels = [1, 0]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)

    def model_init():
        return DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-1, log=True)

        training_args = TrainingArguments(
            output_dir=str(tmp_path.joinpath("results")),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=learning_rate,
            logging_dir=str(tmp_path.joinpath("logs")),
            report_to="none",
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
        )

        train_result = trainer.train()
        return train_result.training_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    best_params = study.best_params

    best_training_args = TrainingArguments(
        output_dir=str(tmp_path.joinpath("results")),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=best_params["learning_rate"],
        logging_dir=str(tmp_path.joinpath("logs")),
    )

    return Trainer(
        model=model,
        args=best_training_args,
        train_dataset=train_dataset,
    )


@pytest.fixture
def transformers_hyperparameter_functional(tmp_path):
    random.seed(555)
    np.random.seed(555)
    torch.manual_seed(555)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_texts = [
        "I simply adore artisinal baked goods!",
        "I thoroughly dislike artisinal bathroom cleaning.",
    ]
    train_labels = [1, 0]
    eval_texts = [
        "It was an excellent experience.",
        "I'd rather pick my teeth with a rusty pitchfork.",
    ]
    eval_labels = [1, 0]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, train_labels)
    eval_dataset = CustomDataset(eval_encodings, eval_labels)

    training_args = TrainingArguments(
        output_dir=str(tmp_path.joinpath("results")),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_dir=str(tmp_path.joinpath("logs")),
        report_to="none",
    )

    def model_init():
        return DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    def my_hp_space_optuna(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        }

    best_run = trainer.hyperparameter_search(
        hp_space=my_hp_space_optuna,
        backend="optuna",
        n_trials=2,
        direction="minimize",
    )

    best_training_args = TrainingArguments(
        output_dir=str(tmp_path.joinpath("best_results")),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=best_run.hyperparameters["learning_rate"],
        logging_dir=str(tmp_path.joinpath("best_logs")),
    )

    return Trainer(
        model=model_init(),
        args=best_training_args,
        train_dataset=train_dataset,
    )


def test_setfit_does_not_autolog(setfit_trainer):
    mlflow.autolog()

    setfit_trainer.train()

    last_run = mlflow.last_active_run()
    assert not last_run
    preds = setfit_trainer.model(
        ["Always carry a towel!", "The hobbits are going to Isengard", "What's tatoes, precious?"]
    )
    assert len(preds) == 3


def test_transformers_trainer_does_not_autolog_sklearn(transformers_trainer):
    mlflow.sklearn.autolog()

    exp = mlflow.set_experiment(experiment_name="trainer_autolog_test")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["_name_or_path"] == "distilbert-base-uncased"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # Checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1


def test_transformers_autolog_adheres_to_global_behavior_using_setfit(setfit_trainer):
    mlflow.transformers.autolog(disable=False)

    setfit_trainer.train()
    assert len(mlflow.search_runs()) == 0
    preds = setfit_trainer.model(["Jim, I'm a doctor, not an archaeologist!"])
    assert len(preds) == 1


def test_transformers_autolog_adheres_to_global_behavior_using_trainer(transformers_trainer):
    mlflow.transformers.autolog()

    exp = mlflow.set_experiment(experiment_name="autolog_with_trainer")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe(["This is pretty ok, I guess", "I came here to chew bubblegum"])
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1


def test_active_autolog_no_setfit_logging_followed_by_successful_sklearn_autolog(
    iris_data, setfit_trainer
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="setfit_with_sklearn")

    # Train and evaluate
    setfit_trainer.train()
    metrics = setfit_trainer.evaluate()
    assert metrics["accuracy"] > 0

    # Run inference
    preds = setfit_trainer.model(
        [
            "i loved the new Star Trek show!",
            "That burger was gross; it tasted like it was made from cat food!",
        ]
    )
    assert len(preds) == 2

    # Test that autologging works for a simple sklearn model (local disabling functions)
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    logged_sklearn_data = mlflow.get_run(run.info.run_id)
    assert logged_sklearn_data.data.tags["estimator_name"] == "KMeans"

    # Assert only the sklearn KMeans model was logged to the experiment

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].info == logged_sklearn_data.info


def test_active_autolog_allows_subsequent_sklearn_autolog(iris_data, transformers_trainer):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="trainer_with_sklearn")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe(["This is pretty ok, I guess", "I came here to chew bubblegum"])
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    logged_sklearn_data = mlflow.get_run(run.info.run_id)
    assert logged_sklearn_data.data.tags["estimator_name"] == "KMeans"

    # Assert only the sklearn KMeans model was logged to the experiment

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 2
    sklearn_run = [x for x in runs if x.info.run_id == run.info.run_id]
    assert sklearn_run[0].info == logged_sklearn_data.info


def test_disabled_sklearn_autologging_does_not_revert_to_enabled_with_setfit(
    iris_data, setfit_trainer
):
    mlflow.autolog()
    mlflow.sklearn.autolog(disable=True)

    exp = mlflow.set_experiment(experiment_name="setfit_with_sklearn_no_autologging")

    # Train and evaluate
    setfit_trainer.train()
    metrics = setfit_trainer.evaluate()
    assert metrics["accuracy"] > 0

    # Run inference
    preds = setfit_trainer.model(
        [
            "i loved the new Star Trek show!",
            "That burger was gross; it tasted like it was made from cat food!",
        ]
    )
    assert len(preds) == 2

    # Test that autologging does not log since it is manually disabled above.
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    # Assert that only the run info is logged
    logged_sklearn_data = mlflow.get_run(run.info.run_id)

    assert logged_sklearn_data.data.params == {}
    assert logged_sklearn_data.data.metrics == {}

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1  # SetFit should not create a run in the experiment
    assert runs[0].info == logged_sklearn_data.info


def test_disable_sklearn_autologging_does_not_revert_with_trainer(iris_data, transformers_trainer):
    mlflow.autolog()
    mlflow.sklearn.autolog(disable=True)

    exp = mlflow.set_experiment(experiment_name="trainer_with_sklearn")

    transformers_trainer.train()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    preds = pipe(
        ["Did you hear that guitar solo? Brilliant!", "That band should avoid playing live."]
    )
    assert len(preds) == 2
    assert all(x["score"] > 0 for x in preds)

    # Test that autologging does not log since it is manually disabled above.
    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        model = sklearn.cluster.KMeans()
        X, y = iris_data
        model.fit(X, y)

    # Assert that only the run info is logged
    logged_sklearn_data = mlflow.get_run(run.info.run_id)

    assert logged_sklearn_data.data.params == {}
    assert logged_sklearn_data.data.metrics == {}

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 2
    sklearn_run = [x for x in runs if x.info.run_id == run.info.run_id]
    assert sklearn_run[0].info == logged_sklearn_data.info


def test_trainer_hyperparameter_tuning_does_not_log_sklearn_model(
    transformers_hyperparameter_trainer,
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="hyperparam_trainer")

    transformers_hyperparameter_trainer.train()
    mlflow.flush_async_logging()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 3.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_hyperparameter_trainer.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1


def test_trainer_hyperparameter_tuning_functional_does_not_log_sklearn_model(
    transformers_hyperparameter_functional,
):
    mlflow.autolog()

    exp = mlflow.set_experiment(experiment_name="hyperparam_trainer_functional")

    transformers_hyperparameter_functional.train()
    mlflow.flush_async_logging()

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["epoch"] == 1.0
    assert last_run.data.params["model_type"] == "distilbert"

    pipe = pipeline(
        task="text-classification",
        model=transformers_hyperparameter_functional.model,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased"),
    )
    assert len(pipe("This is wonderful!")[0]["label"]) > 5  # checking for 'LABEL_0' or 'LABEL_1'

    client = mlflow.MlflowClient()
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 1

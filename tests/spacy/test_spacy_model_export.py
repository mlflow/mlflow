import os
import random
from collections import namedtuple

import pandas as pd
import pytest
import spacy
import yaml
from spacy.util import compounding, minibatch

import mlflow.spacy
from sklearn.datasets import fetch_20newsgroups

from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.conftest import tracking_uri_mock  # pylint: disable=unused-import, E0611

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def spacy_model_with_data():
    # Creating blank model and setting up the spaCy pipeline
    nlp = spacy.blank("en")
    textcat = nlp.create_pipe(
        "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
    )
    nlp.add_pipe(textcat, last=True)

    # Training the model to recognize between computer graphics and baseball in 20newsgroups dataset
    categories = ["comp.graphics", "rec.sport.baseball"]
    for cat in categories:
        textcat.add_label(cat)

    # Split train/test and train the model
    train_x, train_y, test_x, _ = _get_train_test_dataset(categories)
    train_data = list(zip(train_x, [{"cats": cats} for cats in train_y]))
    _train_model(nlp, train_data)
    return ModelWithData(nlp, pd.DataFrame(test_x))


@pytest.fixture
def spacy_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_conda_deps=["pytest"], additional_pip_deps=["spacy"])
    return conda_env


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_model_save_load(spacy_model_with_data, model_path):
    spacy_model = spacy_model_with_data.model
    mlflow.spacy.save_model(spacy_model=spacy_model, path=model_path)
    loaded_model = mlflow.spacy.load_model(model_path)

    # Comparing the meta dictionaries for the original and loaded models
    assert spacy_model.meta == loaded_model.meta

    # Load pyfunc model using saved model and asserting its predictions are equal to the created one
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)
    assert all(
        _predict(spacy_model, spacy_model_with_data.inference_data)
        == pyfunc_loaded.predict(spacy_model_with_data.inference_data)
    )


@pytest.mark.large
def test_model_export_with_schema_and_examples(spacy_model_with_data):
    spacy_model = spacy_model_with_data.model
    signature_ = infer_signature(spacy_model_with_data.inference_data)
    example_ = spacy_model_with_data.inference_data.head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            print(signature is None, example is None)
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.spacy.save_model(
                    spacy_model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


@pytest.mark.large
def test_predict_df_with_wrong_shape(spacy_model_with_data, model_path):
    mlflow.spacy.save_model(spacy_model=spacy_model_with_data.model, path=model_path)
    pyfunc_loaded = mlflow.pyfunc.load_pyfunc(model_path)

    # Concatenating with itself to duplicate column and mess up input shape
    # then asserting n MlflowException is raised
    with pytest.raises(MlflowException):
        pyfunc_loaded.predict(
            pd.concat(
                [spacy_model_with_data.inference_data, spacy_model_with_data.inference_data], axis=1
            )
        )


@pytest.mark.large
def test_model_log(spacy_model_with_data, tracking_uri_mock):  # pylint: disable=unused-argument
    spacy_model = spacy_model_with_data.model
    old_uri = mlflow.get_tracking_uri()
    # should_start_run tests whether or not calling log_model() automatically starts a run.
    for should_start_run in [False, True]:
        with TempDir(chdr=True, remove_on_exit=True):
            try:
                artifact_path = "model"
                if should_start_run:
                    mlflow.start_run()
                mlflow.spacy.log_model(spacy_model=spacy_model, artifact_path=artifact_path)
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
                )

                # Load model
                spacy_model_loaded = mlflow.spacy.load_model(model_uri=model_uri)
                assert all(
                    _predict(spacy_model, spacy_model_with_data.inference_data)
                    == _predict(spacy_model_loaded, spacy_model_with_data.inference_data)
                )
            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(old_uri)


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    spacy_model_with_data, model_path, spacy_custom_env
):
    mlflow.spacy.save_model(
        spacy_model=spacy_model_with_data.model, path=model_path, conda_env=spacy_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spacy_custom_env

    with open(spacy_custom_env, "r") as f:
        spacy_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == spacy_custom_env_text


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(spacy_model_with_data, model_path):
    conda_env = dict(mlflow.spacy.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.spacy.save_model(
        spacy_model=spacy_model_with_data.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    spacy_model_with_data, spacy_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spacy.log_model(
            spacy_model=spacy_model_with_data.model,
            artifact_path=artifact_path,
            conda_env=spacy_custom_env,
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != spacy_custom_env

    with open(spacy_custom_env, "r") as f:
        spacy_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == spacy_custom_env_text


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    spacy_model_with_data, model_path
):
    mlflow.spacy.save_model(
        spacy_model=spacy_model_with_data.model, path=model_path, conda_env=None
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.spacy.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    spacy_model_with_data,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spacy.log_model(spacy_model=spacy_model_with_data.model, artifact_path=artifact_path)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.spacy.get_default_conda_env()


@pytest.mark.large
def test_model_log_with_pyfunc_flavor(spacy_model_with_data):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.spacy.log_model(spacy_model=spacy_model_with_data.model, artifact_path=artifact_path)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

        loaded_model = Model.load(model_path)
        assert pyfunc.FLAVOR_NAME in loaded_model.flavors


@pytest.mark.large
def test_model_log_without_pyfunc_flavor():
    artifact_path = "model"
    nlp = spacy.blank("en")

    # Add a component not compatible with pyfunc
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)

    # Ensure the pyfunc flavor is not present after logging and loading the model
    with mlflow.start_run():
        mlflow.spacy.log_model(spacy_model=nlp, artifact_path=artifact_path)
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

        loaded_model = Model.load(model_path)
        assert loaded_model.flavors.keys() == {"spacy"}


def _train_model(nlp, train_data, n_iter=5):
    optimizer = nlp.begin_training()
    batch_sizes = compounding(4.0, 32.0, 1.001)
    for _ in range(n_iter):
        losses = {}
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)


def _get_train_test_dataset(cats_to_fetch, limit=100):
    newsgroups = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"), shuffle=True, categories=cats_to_fetch
    )
    X = newsgroups.data[:limit]
    y = newsgroups.target[:limit]

    X = [str(x) for x in X]  # Ensure all strings to unicode for python 2.7 compatibility

    # Category 0 comp-graphic, 1 rec.sport baseball. We can threat it as a binary class.
    cats = [{"comp.graphics": not bool(el), "rec.sport.baseball": bool(el)} for el in y]

    split = int(len(X) * 0.8)
    return X[:split], cats[:split], X[split:], cats[split:]


def _predict(spacy_model, test_x):
    return pd.DataFrame(
        {"predictions": test_x.iloc[:, 0].apply(lambda text: spacy_model(text).cats)}
    )

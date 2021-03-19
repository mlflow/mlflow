import os
import pytest
import yaml
from collections import namedtuple
from unittest import mock

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
from fastai.tabular.all import tabular_learner, TabularDataLoaders
from fastai.metrics import accuracy

import mlflow.fastai
import mlflow.utils
from mlflow import pyfunc
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import


ModelWithData = namedtuple("ModelWithData", ["model", "inference_dataframe"])


@pytest.fixture(scope="session")
def fastai_model():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target, name="label")
    dl = TabularDataLoaders.from_df(
        df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
    )
    model = tabular_learner(dl, metrics=accuracy, layers=[3])
    model.fit(1)
    return ModelWithData(model=model, inference_dataframe=X)


@pytest.fixture
def model_path(tmpdir):
    path = os.path.join(str(tmpdir), "model")
    # Seems like Windows absolute path are not currently supported by MLFlow
    # then they are transformed to relative path
    if path.lower().startswith("c:"):
        path = path[2:]
    return path


@pytest.fixture
def fastai_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["fastai", "pytest"])
    return conda_env


@pytest.mark.large
def test_model_save_load(fastai_model, model_path):
    model = fastai_model.model

    mlflow.fastai.save_model(fastai_learner=model, path=model_path)
    reloaded_model = mlflow.fastai.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    # Verify reloaded model computes same predictions as original model
    dl_model = model.dls.test_dl(fastai_model.inference_dataframe)
    dl_reloaded_model = reloaded_model.dls.test_dl(fastai_model.inference_dataframe)

    real_preds, _ = map(
        lambda output: output.numpy() if output is not None else output,
        model.get_preds(dl=dl_model),
    )
    reloaded_preds, _ = map(
        lambda output: output.numpy() if output is not None else output,
        reloaded_model.get_preds(dl=dl_reloaded_model),
    )

    np.testing.assert_array_almost_equal(real_preds, reloaded_preds)

    model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
    reloaded_model_wrapper = mlflow.fastai._FastaiModelWrapper(reloaded_model)

    model_result = model_wrapper.predict(fastai_model.inference_dataframe)
    reloaded_result = reloaded_model_wrapper.predict(fastai_model.inference_dataframe)
    pyfunc_result = reloaded_pyfunc.predict(fastai_model.inference_dataframe)

    np.testing.assert_array_almost_equal(model_result, reloaded_result)
    np.testing.assert_array_almost_equal(reloaded_result, pyfunc_result)


def test_signature_and_examples_are_saved_correctly(fastai_model):
    model = fastai_model.model
    signature_ = infer_signature(fastai_model.inference_dataframe)
    example_ = fastai_model.inference_dataframe.head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.fastai.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(fastai_model, model_path, mock_s3_bucket):
    model = fastai_model.model

    mlflow.fastai.save_model(fastai_learner=fastai_model.model, path=model_path)
    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.fastai.load_model(model_uri=model_uri)

    model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
    reloaded_model_wrapper = mlflow.fastai._FastaiModelWrapper(reloaded_model)

    np.testing.assert_array_almost_equal(
        model_wrapper.predict(fastai_model.inference_dataframe),
        reloaded_model_wrapper.predict(fastai_model.inference_dataframe),
    )


@pytest.mark.large
def test_model_log(fastai_model, model_path):
    old_uri = mlflow.get_tracking_uri()
    model = fastai_model.model
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "model"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["fastai"])

                mlflow.fastai.log_model(
                    fastai_learner=model, artifact_path=artifact_path, conda_env=conda_env
                )

                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
                )

                reloaded_model = mlflow.fastai.load_model(model_uri=model_uri)

                model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
                reloaded_model_wrapper = mlflow.fastai._FastaiModelWrapper(reloaded_model)

                np.testing.assert_array_almost_equal(
                    model_wrapper.predict(fastai_model.inference_dataframe),
                    reloaded_model_wrapper.predict(fastai_model.inference_dataframe),
                )

                model_path = _download_artifact_from_uri(artifact_uri=model_uri)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(old_uri)


def test_log_model_calls_register_model(fastai_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["fastai"])
        mlflow.fastai.log_model(
            fastai_learner=fastai_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(fastai_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["fastai"])
        mlflow.fastai.log_model(
            fastai_learner=fastai_model.model, artifact_path=artifact_path, conda_env=conda_env
        )
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    fastai_model, model_path, fastai_custom_env
):
    mlflow.fastai.save_model(
        fastai_learner=fastai_model.model, path=model_path, conda_env=fastai_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != fastai_custom_env

    with open(fastai_custom_env, "r") as f:
        fastai_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == fastai_custom_env_parsed


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(fastai_model, model_path):
    conda_env = dict(mlflow.fastai.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.fastai.save_model(
        fastai_learner=fastai_model.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    fastai_model, fastai_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.fastai.log_model(
            fastai_learner=fastai_model.model,
            artifact_path=artifact_path,
            conda_env=fastai_custom_env,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != fastai_custom_env

    with open(fastai_custom_env, "r") as f:
        fastai_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == fastai_custom_env_parsed


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    fastai_model, model_path
):
    mlflow.fastai.save_model(fastai_learner=fastai_model.model, path=model_path, conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.fastai.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    fastai_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.fastai.log_model(
            fastai_learner=fastai_model.model, artifact_path=artifact_path, conda_env=None
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.fastai.get_default_conda_env()

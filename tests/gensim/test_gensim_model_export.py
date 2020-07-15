from __future__ import print_function

import mock
import os
import pytest
import yaml
from collections import namedtuple

import mlflow.gensim
import mlflow.utils
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import score_model_in_sagemaker_docker_container

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="session")
def gensim_word2vec_model():
    corpus = api.load('text8')
    model = Word2Vec(corpus)

    return ModelWithData(model=model, inference_data=corpus)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def gensim_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
            conda_env,
            additional_conda_deps=["gensim", "pytest"])
    return conda_env


@pytest.mark.large
def test_model_save_load(gensim_word2vec_model, model_path):
    word2vecmodel = gensim_word2vec_model.model

    mlflow.gensim.save_model(gen_model=word2vecmodel, path=model_path)
    reloaded_word2vec_model = mlflow.gensim.load_model(model_uri=model_path)

    assert all([a == b for a, b in zip(word2vecmodel.wv.most_similar('tree'),
                                       reloaded_word2vec_model.wv.most_similar('tree'))])


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(gensim_word2vec_model, model_path, mock_s3_bucket):

    mlflow.gensim.save_model(gen_model=gensim_word2vec_model.model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_word2vec_model = mlflow.gensim.load_model(model_uri=model_uri)

    assert all([a == b for a, b in zip(gensim_word2vec_model.model.wv.most_similar('tree'),
                                       reloaded_word2vec_model.wv.most_similar('tree'))])


@pytest.mark.large
def test_model_log(gensim_word2vec_model):

    old_uri = mlflow.get_tracking_uri()
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "embedding"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["gensim"])

                mlflow.gensim.log_model(
                        gen_model=gensim_word2vec_model.model,
                        artifact_path=artifact_path,
                        conda_env=conda_env)
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id,
                    artifact_path=artifact_path)

                reloaded_word2vec_model = mlflow.gensim.load_model(model_uri=model_uri)
                assert all(
                    [a == b for a, b in zip(gensim_word2vec_model.model.wv.most_similar('tree'),
                                            reloaded_word2vec_model.wv.most_similar('tree'))]
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


def test_log_model_calls_register_model(gensim_word2vec_model):
    artifact_path = "embedding"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["gensim"])
        mlflow.gensim.log_model(gen_model=gensim_word2vec_model.model, artifact_path=artifact_path,
                                conda_env=conda_env, registered_model_name="AdsModel1")

        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id,
                                                            artifact_path=artifact_path)
        mlflow.register_model.assert_called_once_with(model_uri, "AdsModel1")


def test_log_model_no_registered_model_name(gensim_word2vec_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["gensim"])
        mlflow.gensim.log_model(gen_model=gensim_word2vec_model.model, artifact_path=artifact_path,
                                conda_env=conda_env)
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        gensim_word2vec_model, model_path, gensim_custom_env):

    mlflow.gensim.save_model(
            gen_model=gensim_word2vec_model.model, path=model_path, conda_env=gensim_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != gensim_custom_env

    with open(gensim_custom_env, "r") as f:
        gensim_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == gensim_custom_env_parsed


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(gensim_word2vec_model, model_path):
    conda_env = dict(mlflow.gensim.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.gensim.save_model(
            gen_model=gensim_word2vec_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
        gensim_word2vec_model, gensim_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.gensim.log_model(gen_model=gensim_word2vec_model.model,
                                artifact_path=artifact_path,
                                conda_env=gensim_custom_env)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path=artifact_path)

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != gensim_custom_env

    with open(gensim_custom_env, "r") as f:
        gensim_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == gensim_custom_env_parsed


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        gensim_word2vec_model, model_path):
    word2vec_model = gensim_word2vec_model.model
    mlflow.gensim.save_model(gen_model=word2vec_model, path=model_path, conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.gensim.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        gensim_word2vec_model):
    artifact_path = "model"
    word2vec_model = gensim_word2vec_model.model
    with mlflow.start_run():
        mlflow.gensim.log_model(gen_model=word2vec_model,
                                artifact_path=artifact_path,
                                conda_env=None)

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path=artifact_path)

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.gensim.get_default_conda_env()

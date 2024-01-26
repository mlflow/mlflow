"""
This test class is used for comprehensive testing of serving docker images for all MLflow flavors.
As such, it is not intended to be run on a regular basis and is skipped by default. Rather, it
should be run manually when making changes to the core docker logic.

To run this test, run the following command to disable the skip flag:

    $ pytest tests/pyfunc/test_docker_flavors.py --no-skips

"""

import json
import os
import time
from operator import itemgetter

import pandas as pd
import pytest
import requests
from packaging.version import Version
from sklearn import datasets

import docker
import mlflow
from mlflow.models.flavor_backend_registry import get_flavor_backend

from tests.catboost.test_catboost_model_export import reg_model
from tests.diviner.test_diviner_model_export import (
    diviner_data,
    diviner_groups,
    grouped_prophet,
)
from tests.fastai.test_fastai_model_export import fastai_model as fastai_model_raw
from tests.h2o.test_h2o_model_export import h2o_iris_model
from tests.helper_functions import get_safe_port
from tests.langchain.test_langchain_model_export import fake_chat_model
from tests.lightgbm.test_lightgbm_model_export import lgb_model
from tests.paddle.test_paddle_model_export import pd_model
from tests.pmdarima.test_pmdarima_model_export import (
    auto_arima_object_model,
    pmdarima_test_data,
)
from tests.prophet.test_prophet_model_export import prophet_model as prophet_raw_model
from tests.sklearn.test_sklearn_model_export import iris_df, sklearn_knn_model
from tests.spacy.test_spacy_model_export import spacy_model_with_data
from tests.spark.test_spark_model_export import (
    iris_spark_and_pandas_df,
    spark,
    spark_model_iris,
)
from tests.statsmodels.model_fixtures import ols_model
from tests.tensorflow.test_keras_model_export import get_model
from tests.tensorflow.test_tensorflow2_core_model_export import tf2_toy_model
from tests.transformers.helper import load_small_seq2seq_pipeline

_TEST_IMAGE_NAME = "test_image"
_MLFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

docker_client = docker.from_env()


def get_released_mlflow_version():
    url = "https://pypi.org/pypi/mlflow/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    versions = [Version(v) for v in data["releases"].keys() if not v.endswith(("a", "b", "rc"))]
    return sorted(versions, reverse=True)[0]


_LATEST_MLFLOW_VERSION = get_released_mlflow_version()


def save_model_with_latest_mlflow_version(flavor, **kwargs):
    """
    Save a model with overriding MLflow version from dev version to the latest released version.
    By default a model is saved with the dev version of MLflow, which is not available on PyPI.
    Usually we can be workaround this by adding --serve-wheel flag that starts local PyPI server,
    however, this doesn't work when installing dependencies inside Docker container. Hence, this
    function uses `extra_pip_requirements` to save the model with the latest released MLflow.
    """
    if flavor == "langchain":
        kwargs["pip_requirements"] = [f"mlflow[gateway]=={_LATEST_MLFLOW_VERSION}", "langchain"]
    else:
        kwargs["extra_pip_requirements"] = [f"mlflow=={_LATEST_MLFLOW_VERSION}"]
    flavor_module = getattr(mlflow, flavor)
    flavor_module.save_model(**kwargs)


@pytest.fixture(autouse=True)
def clean_up_docker_image():
    yield

    try:
        # Get all containers using the test image
        containers = docker_client.containers.list(filters={"ancestor": _TEST_IMAGE_NAME})
        for container in containers:
            container.remove(force=True)

        # Clean up the image
        docker_client.images.remove(_TEST_IMAGE_NAME, force=True)
    except Exception:
        pass


# TODO: Add skipif condition for normal run and make it opt-in
@pytest.mark.parametrize(
    ("flavor"),
    [
        "catboost",
        "diviner",
        "fastai",
        "h2o",
        # "johnsnowlabs", # Couldn't test JohnSnowLab locally due to license issue
        "keras",
        "langchain",
        "lightgbm",
        # "mleap", # Mleap model logging is deprecated since 2.6.1
        "onnx",
        # "openai", # OPENAI API KEY is not necessarily available for everyone
        "paddle",
        "pmdarima",
        "prophet",
        "pyfunc",
        "pytorch",
        "sklearn",
        "spacy",
        "spark",
        "statsmodels",
        "tensorflow",
        "transformers",
    ],
)
def test_build_image_and_serve(flavor, request):
    model_path = request.getfixturevalue(f"{flavor}_model")

    # Make a scoring request
    with open(os.path.join(model_path, "input_example.json")) as f:
        input_example = json.load(f)

    if "columns" in input_example or "data" in input_example:
        input_example = {"dataframe_split": input_example}

    # Build an image
    backend = get_flavor_backend(model_uri=model_path, docker_build=True)
    backend.build_image(
        model_uri=model_path,
        image_name=_TEST_IMAGE_NAME,
        mlflow_home=_MLFLOW_ROOT,  # Required to prevent installing dev version of MLflow from PyPI
    )

    # Run a container
    port = get_safe_port()
    docker_client.containers.run(
        image=_TEST_IMAGE_NAME,
        ports={8080: port},
        detach=True,
    )

    # Wait until the container to start
    timeout = 60
    start_time = time.time()
    success = False
    while time.time() < start_time + timeout:
        try:
            # Send health check
            response = requests.get(url=f"http://localhost:{port}/ping")
            if response.status_code == 200:
                success = True
                break
        except Exception:
            time.sleep(5)

    if not success:
        raise TimeoutError("TBA")

    # Make a scoring request with a saved input example
    with open(os.path.join(model_path, "input_example.json")) as f:
        input_example = json.load(f)

    # Wrap Pandas dataframe in a proper payload format
    if "columns" in input_example or "data" in input_example:
        input_example = {"dataframe_split": input_example}

    response = requests.post(
        url=f"http://localhost:{port}/invocations",
        data=json.dumps(input_example),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200, f"Response: {response.text}"
    assert "predictions" in response.json(), f"Response: {response.text}"


@pytest.fixture
def catboost_model(tmp_path, reg_model):
    model_path = str(tmp_path / "catboost_model")
    save_model_with_latest_mlflow_version(
        flavor="catboost",
        cb_model=reg_model.model,
        path=model_path,
        input_example=reg_model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def diviner_model(tmp_path, grouped_prophet, diviner_groups):
    model_path = str(tmp_path / "diviner_model")

    save_model_with_latest_mlflow_version(
        flavor="diviner",
        diviner_model=grouped_prophet,
        path=model_path,
        input_example=pd.DataFrame(
            {"groups": [diviner_groups], "horizon": 10, "frequency": "D"}, index=[0]
        ),
    )
    return model_path


@pytest.fixture
def fastai_model(tmp_path, fastai_model_raw):
    model_path = str(tmp_path / "fastai_model")
    save_model_with_latest_mlflow_version(
        flavor="fastai",
        fastai_learner=fastai_model_raw.model,
        path=model_path,
        input_example=fastai_model_raw.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def h2o_model(tmp_path, h2o_iris_model):
    model_path = str(tmp_path / "h2o_model")
    save_model_with_latest_mlflow_version(
        flavor="h2o",
        h2o_model=h2o_iris_model.model,
        path=model_path,
        input_example=h2o_iris_model.inference_data.as_data_frame()[:1],
    )
    return model_path


@pytest.fixture
def keras_model(tmp_path):
    model_path = str(tmp_path / "keras_model")
    iris_data = datasets.load_iris(return_X_y=True)
    save_model_with_latest_mlflow_version(
        flavor="tensorflow",
        model=get_model(iris_data),
        path=model_path,
        input_example=iris_data[0][:1],
    )
    return model_path


@pytest.fixture
def langchain_model(tmp_path):
    from langchain.schema.runnable import RunnablePassthrough

    chain = RunnablePassthrough() | itemgetter("messages")
    model_path = str(tmp_path / "langchain_model")
    save_model_with_latest_mlflow_version(
        flavor="langchain", lc_model=chain, path=model_path, input_example={"messages": "Hi"}
    )
    return model_path


@pytest.fixture
def lightgbm_model(tmp_path, lgb_model):
    model_path = str(tmp_path / "lightgbm_model")
    save_model_with_latest_mlflow_version(
        flavor="lightgbm",
        lgb_model=lgb_model.model,
        path=model_path,
        input_example=lgb_model.inference_dataframe.to_numpy()[:1],
    )
    return model_path


@pytest.fixture
def onnx_model(tmp_path):
    import numpy as np
    import onnx
    import torch
    from torch import nn

    model = torch.nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    onnx_model_path = os.path.join(tmp_path, "torch_onnx")
    torch.onnx.export(
        model,
        torch.randn(1, 4),
        onnx_model_path,
        dynamic_axes={"input": {0: "batch"}},
        input_names=["input"],
    )
    onnx_model = onnx.load(onnx_model_path)

    model_path = str(tmp_path / "onnx_model")
    save_model_with_latest_mlflow_version(
        flavor="onnx",
        onnx_model=onnx_model,
        path=model_path,
        input_example=np.random.rand(1, 4).astype(np.float32),
    )
    return model_path


@pytest.fixture
def paddle_model(tmp_path, pd_model):
    model_path = str(tmp_path / "paddle_model")
    save_model_with_latest_mlflow_version(
        flavor="paddle",
        pd_model=pd_model.model,
        path=model_path,
        input_example=pd_model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def pmdarima_model(tmp_path, auto_arima_object_model):
    model_path = str(tmp_path / "pmdarima_model")
    save_model_with_latest_mlflow_version(
        flavor="pmdarima",
        pmdarima_model=auto_arima_object_model,
        path=model_path,
        input_example=pd.DataFrame({"n_periods": [30]}),
    )
    return model_path


@pytest.fixture
def prophet_model(tmp_path, prophet_raw_model):
    model_path = str(tmp_path / "prophet_model")
    save_model_with_latest_mlflow_version(
        flavor="prophet",
        pr_model=prophet_raw_model.model,
        path=model_path,
        input_example=prophet_raw_model.data[:1],
    )
    return model_path


@pytest.fixture
def pyfunc_model(tmp_path):
    class CustomModel(mlflow.pyfunc.PythonModel):
        def __init__(self):
            pass

        def predict(self, context, model_input):
            return model_input

    model_path = str(tmp_path / "pyfunc_model")
    save_model_with_latest_mlflow_version(
        flavor="pyfunc",
        python_model=CustomModel(),
        path=model_path,
        input_example=[1, 2, 3],
    )
    return model_path


@pytest.fixture
def pytorch_model(tmp_path):
    from torch import nn, randn

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    model_path = str(tmp_path / "pytorch_model")
    save_model_with_latest_mlflow_version(
        flavor="pytorch",
        pytorch_model=model,
        path=model_path,
        input_example=randn(1, 4).numpy(),
    )
    return model_path


@pytest.fixture
def sklearn_model(tmp_path, sklearn_knn_model):
    model_path = str(tmp_path / "sklearn_model")
    save_model_with_latest_mlflow_version(
        flavor="sklearn",
        sk_model=sklearn_knn_model.model,
        path=model_path,
        input_example=sklearn_knn_model.inference_data[:1],
    )
    return model_path


@pytest.fixture
def spacy_model(tmp_path, spacy_model_with_data):
    model_path = str(tmp_path / "spacy_model")
    save_model_with_latest_mlflow_version(
        flavor="spacy",
        spacy_model=spacy_model_with_data.model,
        path=model_path,
        input_example=spacy_model_with_data.inference_data[:1],
    )
    return model_path


@pytest.fixture
def spark_model(tmp_path, spark_model_iris):
    model_path = str(tmp_path / "spark_model")
    save_model_with_latest_mlflow_version(
        flavor="spark",
        spark_model=spark_model_iris.model,
        path=model_path,
        input_example=spark_model_iris.spark_df.toPandas()[:1],
    )
    return model_path


@pytest.fixture
def statsmodels_model(tmp_path):
    model = ols_model()
    model_path = str(tmp_path / "statsmodels_model")
    save_model_with_latest_mlflow_version(
        flavor="statsmodels",
        statsmodels_model=model.model,
        path=model_path,
        input_example=model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def tensorflow_model(tmp_path, tf2_toy_model):
    model_path = str(tmp_path / "tensorflow_model")
    save_model_with_latest_mlflow_version(
        flavor="tensorflow",
        model=tf2_toy_model.model,
        path=model_path,
        input_example=tf2_toy_model.inference_data[:1],
    )
    return model_path


@pytest.fixture
def transformers_model(tmp_path):
    pipeline = load_small_seq2seq_pipeline()
    model_path = str(tmp_path / "transformers_model")
    save_model_with_latest_mlflow_version(
        flavor="transformers",
        transformers_model=pipeline,
        path=model_path,
        input_example="hi",
    )
    return model_path

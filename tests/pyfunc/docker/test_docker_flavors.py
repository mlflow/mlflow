"""
This test class is used for comprehensive testing of serving docker images for all MLflow flavors.
As such, it is not intended to be run on a regular basis and is skipped by default. Rather, it
should be run manually when making changes to the core docker logic.

To run this test, run the following command manually

    $ pytest tests/pyfunc/test_docker_flavors.py

"""

import json
import os
import shutil
import time
from operator import itemgetter

import pandas as pd
import pytest
import requests

import mlflow
from mlflow.environment_variables import _MLFLOW_RUN_SLOW_TESTS
from mlflow.models.flavor_backend_registry import get_flavor_backend

# Only import model fixtures if when MLFLOW_RUN_SLOW_TESTS environment variable is set to true
if _MLFLOW_RUN_SLOW_TESTS.get():
    from tests.catboost.test_catboost_model_export import reg_model  # noqa: F401
    from tests.diviner.test_diviner_model_export import (  # noqa: F401
        diviner_data,
        diviner_groups,
        grouped_prophet,
    )
    from tests.fastai.test_fastai_model_export import fastai_model as fastai_model_raw  # noqa: F401
    from tests.h2o.test_h2o_model_export import h2o_iris_model  # noqa: F401
    from tests.helper_functions import get_safe_port
    from tests.langchain.test_langchain_model_export import fake_chat_model  # noqa: F401
    from tests.lightgbm.test_lightgbm_model_export import lgb_model  # noqa: F401
    from tests.models.test_model import iris_data, sklearn_knn_model  # noqa: F401
    from tests.paddle.test_paddle_model_export import pd_model  # noqa: F401
    from tests.pmdarima.test_pmdarima_model_export import (  # noqa: F401
        auto_arima_object_model,
        test_data,
    )
    from tests.prophet.test_prophet_model_export import (
        prophet_model as prophet_raw_model,  # noqa: F401
    )
    from tests.pyfunc.docker.conftest import (
        MLFLOW_ROOT,
        TEST_IMAGE_NAME,
        docker_client,
        save_model_with_latest_mlflow_version,
    )
    from tests.spacy.test_spacy_model_export import spacy_model_with_data  # noqa: F401
    from tests.spark.test_spark_model_export import (  # noqa: F401
        iris_df,
        spark,
        spark_model_iris,
    )
    from tests.statsmodels.model_fixtures import ols_model
    from tests.tensorflow.test_tensorflow2_core_model_export import tf2_toy_model  # noqa: F401
    from tests.transformers.helper import load_small_qa_pipeline, load_small_seq2seq_pipeline


pytestmark = pytest.mark.skipif(
    not _MLFLOW_RUN_SLOW_TESTS.get(),
    reason="Skip slow tests. Set MLFLOW_RUN_SLOW_TESTS environment variable to run them.",
)


@pytest.fixture
def model_path(tmp_path):
    model_path = tmp_path.joinpath("model")

    yield model_path

    # Pytest keeps the temporary directory created by `tmp_path` fixture for 3 recent test sessions
    # by default. This is useful for debugging during local testing, but in CI it just wastes the
    # disk space.
    if os.getenv("GITHUB_ACTIONS") == "true":
        shutil.rmtree(model_path, ignore_errors=True)


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
        "transformers_pt",  # Test with Pytorch-based model
        "transformers_tf",  # Test with TensorFlow-based model
    ],
)
def test_build_image_and_serve(flavor, request):
    model_path = str(request.getfixturevalue(f"{flavor}_model"))
    flavor = flavor.split("_")[0]  # Remove _pt or _tf from the flavor name

    # Build an image
    backend = get_flavor_backend(model_uri=model_path, docker_build=True)
    backend.build_image(
        model_uri=model_path,
        image_name=TEST_IMAGE_NAME,
        mlflow_home=MLFLOW_ROOT,  # Required to prevent installing dev version of MLflow from PyPI
    )

    # Run a container
    port = get_safe_port()
    docker_client.containers.run(
        image=TEST_IMAGE_NAME,
        ports={8080: port},
        detach=True,
    )

    # Wait for the container to start
    for _ in range(30):
        try:
            response = requests.get(url=f"http://localhost:{port}/ping")
            if response.ok:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(5)
    else:
        raise TimeoutError("Failed to start server.")

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
def catboost_model(model_path, reg_model):
    save_model_with_latest_mlflow_version(
        flavor="catboost",
        cb_model=reg_model.model,
        path=model_path,
        input_example=reg_model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def diviner_model(model_path, grouped_prophet, diviner_groups):
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
def fastai_model(model_path, fastai_model_raw):
    save_model_with_latest_mlflow_version(
        flavor="fastai",
        fastai_learner=fastai_model_raw.model,
        path=model_path,
        input_example=fastai_model_raw.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def h2o_model(model_path, h2o_iris_model):
    save_model_with_latest_mlflow_version(
        flavor="h2o",
        h2o_model=h2o_iris_model.model,
        path=model_path,
        input_example=h2o_iris_model.inference_data.as_data_frame()[:1],
    )
    return model_path


@pytest.fixture
def keras_model(model_path, iris_data):
    from sklearn import datasets
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(3, input_dim=4))
    model.add(Dense(1))

    X, y = datasets.load_iris(return_X_y=True)
    save_model_with_latest_mlflow_version(
        flavor="tensorflow",
        model=model,
        path=model_path,
        input_example=X[:3, :],
    )
    return model_path


@pytest.fixture
def langchain_model(model_path):
    from langchain.schema.runnable import RunnablePassthrough

    chain = RunnablePassthrough() | itemgetter("messages")
    save_model_with_latest_mlflow_version(
        flavor="langchain", lc_model=chain, path=model_path, input_example={"messages": "Hi"}
    )
    return model_path


@pytest.fixture
def lightgbm_model(model_path, lgb_model):
    save_model_with_latest_mlflow_version(
        flavor="lightgbm",
        lgb_model=lgb_model.model,
        path=model_path,
        input_example=lgb_model.inference_dataframe.to_numpy()[:1],
    )
    return model_path


@pytest.fixture
def onnx_model(tmp_path, model_path):
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
def paddle_model(model_path, pd_model):
    save_model_with_latest_mlflow_version(
        flavor="paddle",
        pd_model=pd_model.model,
        path=model_path,
        input_example=pd_model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def pmdarima_model(model_path, auto_arima_object_model):
    save_model_with_latest_mlflow_version(
        flavor="pmdarima",
        pmdarima_model=auto_arima_object_model,
        path=model_path,
        input_example=pd.DataFrame({"n_periods": [30]}),
    )
    return model_path


@pytest.fixture
def prophet_model(model_path, prophet_raw_model):
    save_model_with_latest_mlflow_version(
        flavor="prophet",
        pr_model=prophet_raw_model.model,
        path=model_path,
        input_example=prophet_raw_model.data[:1],
    )
    return model_path


@pytest.fixture
def pyfunc_model(model_path):
    class CustomModel(mlflow.pyfunc.PythonModel):
        def __init__(self):
            pass

        def predict(self, context, model_input):
            return model_input

    save_model_with_latest_mlflow_version(
        flavor="pyfunc",
        python_model=CustomModel(),
        path=model_path,
        input_example=[1, 2, 3],
    )
    return model_path


@pytest.fixture
def pytorch_model(model_path):
    from torch import nn, randn

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    save_model_with_latest_mlflow_version(
        flavor="pytorch",
        pytorch_model=model,
        path=model_path,
        input_example=randn(1, 4).numpy(),
    )
    return model_path


@pytest.fixture
def sklearn_model(model_path, sklearn_knn_model, iris_data):
    save_model_with_latest_mlflow_version(
        flavor="sklearn",
        sk_model=sklearn_knn_model,
        path=model_path,
        input_example=iris_data[0][:1],
    )
    return model_path


@pytest.fixture
def spacy_model(model_path, spacy_model_with_data):
    save_model_with_latest_mlflow_version(
        flavor="spacy",
        spacy_model=spacy_model_with_data.model,
        path=model_path,
        input_example=spacy_model_with_data.inference_data[:1],
    )
    return model_path


@pytest.fixture
def spark_model(model_path, spark_model_iris):
    save_model_with_latest_mlflow_version(
        flavor="spark",
        spark_model=spark_model_iris.model,
        path=model_path,
        input_example=spark_model_iris.spark_df.toPandas()[:1],
    )
    return model_path


@pytest.fixture
def statsmodels_model(model_path):
    model = ols_model()
    save_model_with_latest_mlflow_version(
        flavor="statsmodels",
        statsmodels_model=model.model,
        path=model_path,
        input_example=model.inference_dataframe[:1],
    )
    return model_path


@pytest.fixture
def tensorflow_model(model_path, tf2_toy_model):
    save_model_with_latest_mlflow_version(
        flavor="tensorflow",
        model=tf2_toy_model.model,
        path=model_path,
        input_example=tf2_toy_model.inference_data[:1],
    )
    return model_path


@pytest.fixture
def transformers_pt_model(model_path):
    pipeline = load_small_seq2seq_pipeline()
    save_model_with_latest_mlflow_version(
        flavor="transformers",
        transformers_model=pipeline,
        path=model_path,
        input_example="hi",
    )
    return model_path


@pytest.fixture
def transformers_tf_model(model_path):
    pipeline = load_small_qa_pipeline()
    save_model_with_latest_mlflow_version(
        flavor="transformers",
        transformers_model=pipeline,
        path=model_path,
        input_example={"question": "What is MLflow", "context": "It's an open source platform"},
    )
    return model_path

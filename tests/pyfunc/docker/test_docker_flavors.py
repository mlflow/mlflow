"""
This test class is used for comprehensive testing of serving docker images for all MLflow flavors.
As such, it is not intended to be run on a regular basis and is skipped by default. Rather, it
should be run manually when making changes to the core docker logic.

To run this test, run the following command manually

    $ pytest tests/pyfunc/test_docker_flavors.py

"""

import json
import os
import time

from packaging.version import Version
import pytest
import requests

from mlflow.models.flavor_backend_registry import get_flavor_backend
from tests.helper_functions import get_safe_port
from tests.pyfunc.docker.conftest import MLFLOW_ROOT, TEST_IMAGE_NAME, docker_client


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Time consuming tests")
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

    # Wait until the container to start
    timeout = 120
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

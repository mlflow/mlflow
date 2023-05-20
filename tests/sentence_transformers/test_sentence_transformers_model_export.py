import numpy as np
from sentence_transformers import SentenceTransformer
import pytest

import mlflow


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")

@pytest.fixture
def basic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# def test_dependency_inference():



def test_model_save_and_load(model_path, basic_model):

    mlflow.sentence_transformers.save_model(
        model=basic_model,
        path=model_path
    )

    loaded_model = mlflow.sentence_transformers.load_model(model_path)

    encoded_single = loaded_model.encode("I'm just a simple string; nothing to see here.")
    encoded_multi = loaded_model.encode(["I'm a string", "I'm also a string", "Please encode me"])

    assert isinstance(encoded_single, np.ndarray)
    assert len(encoded_single) == 384
    assert isinstance(encoded_multi, np.ndarray)
    assert len(encoded_multi) == 3
    assert all(len(x) == 384 for x in encoded_multi)


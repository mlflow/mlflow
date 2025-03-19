import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.artifacts import (
    CsvEvaluationArtifact,
    ImageEvaluationArtifact,
    JsonEvaluationArtifact,
    NumpyEvaluationArtifact,
    ParquetEvaluationArtifact,
    PickleEvaluationArtifact,
    TextEvaluationArtifact,
    _infer_artifact_type_and_ext,
)
from mlflow.models.evaluation.default_evaluator import _CustomArtifact


@pytest.fixture
def cm_fn_tuple():
    return _CustomArtifact(lambda: None, "", 0, "")


def __generate_dummy_json_file(path):
    with open(path, "w") as f:
        json.dump([1, 2, 3], f)


class __DummyClass:
    def __init__(self):
        self.test = 1


@pytest.mark.parametrize(
    ("is_file", "artifact", "artifact_type", "ext"),
    [
        (True, lambda path: Figure().savefig(path), ImageEvaluationArtifact, "png"),
        (True, lambda path: Figure().savefig(path), ImageEvaluationArtifact, "jpg"),
        (True, lambda path: Figure().savefig(path), ImageEvaluationArtifact, "jpeg"),
        (True, __generate_dummy_json_file, JsonEvaluationArtifact, "json"),
        (True, lambda path: pathlib.Path(path).write_text("test"), TextEvaluationArtifact, "txt"),
        (
            True,
            lambda path: np.save(path, np.array([1, 2, 3]), allow_pickle=False),
            NumpyEvaluationArtifact,
            "npy",
        ),
        (
            True,
            lambda path: pd.DataFrame({"test": [1, 2, 3]}).to_csv(path, index=False),
            CsvEvaluationArtifact,
            "csv",
        ),
        (
            True,
            lambda path: pd.DataFrame({"test": [1, 2, 3]}).to_parquet(path),
            ParquetEvaluationArtifact,
            "parquet",
        ),
        (False, pd.DataFrame({"test": [1, 2, 3]}), CsvEvaluationArtifact, "csv"),
        (False, np.array([1, 2, 3]), NumpyEvaluationArtifact, "npy"),
        (False, Figure(), ImageEvaluationArtifact, "png"),
        (False, {"a": 1, "b": "e", "c": 1.2, "d": [1, 2]}, JsonEvaluationArtifact, "json"),
        (False, [1, 2, 3, "test"], JsonEvaluationArtifact, "json"),
        (False, '{"a": 1, "b": [1.2, 3]}', JsonEvaluationArtifact, "json"),
        (False, '[1, 2, 3, "test"]', JsonEvaluationArtifact, "json"),
        (False, __DummyClass(), PickleEvaluationArtifact, "pickle"),
    ],
)
def test_infer_artifact_type_and_ext(is_file, artifact, artifact_type, ext, tmp_path, cm_fn_tuple):
    if is_file:
        artifact_representation = tmp_path / f"test.{ext}"
        artifact(artifact_representation)
    else:
        artifact_representation = artifact
    inferred_from_path, inferred_type, inferred_ext = _infer_artifact_type_and_ext(
        f"{ext}_{artifact_type.__name__}_artifact", artifact_representation, cm_fn_tuple
    )
    assert not is_file ^ inferred_from_path
    assert inferred_type is artifact_type
    assert inferred_ext == f".{ext}"


def test_infer_artifact_type_and_ext_raise_exception_for_non_file_non_json_str(cm_fn_tuple):
    with pytest.raises(
        MlflowException,
        match="with string representation 'some random str' that is "
        "neither a valid path to a file nor a JSON string",
    ):
        _infer_artifact_type_and_ext("test_artifact", "some random str", cm_fn_tuple)


def test_infer_artifact_type_and_ext_raise_exception_for_non_existent_path(tmp_path, cm_fn_tuple):
    path = tmp_path / "does_not_exist_path"
    with pytest.raises(MlflowException, match=f"with path '{path}' does not exist"):
        _infer_artifact_type_and_ext("test_artifact", path, cm_fn_tuple)


def test_infer_artifact_type_and_ext_raise_exception_for_non_file_artifact(tmp_path, cm_fn_tuple):
    with pytest.raises(MlflowException, match=f"with path '{tmp_path}' is not a file"):
        _infer_artifact_type_and_ext("non_file_artifact", tmp_path, cm_fn_tuple)


def test_infer_artifact_type_and_ext_raise_exception_for_unsupported_ext(tmp_path, cm_fn_tuple):
    path = tmp_path / "invalid_ext_example.some_ext"
    with open(path, "w") as f:
        f.write("some stuff that shouldn't be read")
    with pytest.raises(
        MlflowException,
        match=f"with path '{path}' does not match any of the supported file extensions",
    ):
        _infer_artifact_type_and_ext("invalid_ext_artifact", path, cm_fn_tuple)

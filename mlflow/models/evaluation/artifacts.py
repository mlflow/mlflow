import json
import pathlib
import pickle
from json import JSONDecodeError
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationArtifact
from mlflow.utils.annotations import developer_stable
from mlflow.utils.proto_json_utils import NumpyEncoder


@developer_stable
class ImageEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.save(output_artifact_path)

    def _load_content_from_file(self, local_artifact_path):
        from PIL.Image import open as open_image

        self._content = open_image(local_artifact_path)
        self._content.load()  # Load image and close the file descriptor.
        return self._content


@developer_stable
class CsvEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content


@developer_stable
class ParquetEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.to_parquet(output_artifact_path, compression="brotli")

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_parquet(local_artifact_path)
        return self._content


@developer_stable
class NumpyEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        np.save(output_artifact_path, self._content, allow_pickle=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = np.load(local_artifact_path, allow_pickle=False)
        return self._content


@developer_stable
class JsonEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        with open(output_artifact_path, "w") as f:
            json.dump(self._content, f)

    def _load_content_from_file(self, local_artifact_path):
        with open(local_artifact_path) as f:
            self._content = json.load(f)
        return self._content


@developer_stable
class TextEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        with open(output_artifact_path, "w") as f:
            f.write(self._content)

    def _load_content_from_file(self, local_artifact_path):
        with open(local_artifact_path) as f:
            self._content = f.read()
        return self._content


@developer_stable
class PickleEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        with open(output_artifact_path, "wb") as f:
            pickle.dump(self._content, f)

    def _load_content_from_file(self, local_artifact_path):
        with open(local_artifact_path, "rb") as f:
            self._content = pickle.load(f)
        return self._content


_EXT_TO_ARTIFACT_MAP = {
    ".png": ImageEvaluationArtifact,
    ".jpg": ImageEvaluationArtifact,
    ".jpeg": ImageEvaluationArtifact,
    ".json": JsonEvaluationArtifact,
    ".npy": NumpyEvaluationArtifact,
    ".csv": CsvEvaluationArtifact,
    ".parquet": ParquetEvaluationArtifact,
    ".txt": TextEvaluationArtifact,
}

_TYPE_TO_EXT_MAP = {
    pd.DataFrame: ".csv",
    np.ndarray: ".npy",
    plt.Figure: ".png",
}

_TYPE_TO_ARTIFACT_MAP = {
    pd.DataFrame: CsvEvaluationArtifact,
    np.ndarray: NumpyEvaluationArtifact,
    plt.Figure: ImageEvaluationArtifact,
}


class _InferredArtifactProperties(NamedTuple):
    from_path: bool
    type: type[EvaluationArtifact]
    ext: str


def _infer_artifact_type_and_ext(artifact_name, raw_artifact, custom_metric_tuple):
    """
    This function performs type and file extension inference on the provided artifact

    Args:
        artifact_name: The name of the provided artifact
        raw_artifact: The artifact object
        custom_metric_tuple: Containing a user provided function and its index in the
            ``custom_metrics`` parameter of ``mlflow.evaluate``

    Returns:
        InferredArtifactProperties namedtuple
    """

    exception_header = (
        f"Custom metric function '{custom_metric_tuple.name}' at index "
        f"{custom_metric_tuple.index} in the `custom_metrics` parameter produced an "
        f"artifact '{artifact_name}'"
    )

    # Given a string, first see if it is a path. Otherwise, check if it is a JsonEvaluationArtifact
    if isinstance(raw_artifact, str):
        potential_path = pathlib.Path(raw_artifact)
        if potential_path.exists():
            raw_artifact = potential_path
        else:
            try:
                json.loads(raw_artifact)
                return _InferredArtifactProperties(
                    from_path=False, type=JsonEvaluationArtifact, ext=".json"
                )
            except JSONDecodeError:
                raise MlflowException(
                    f"{exception_header} with string representation '{raw_artifact}' that is "
                    f"neither a valid path to a file nor a JSON string."
                )

    # Type inference based on the file extension
    if isinstance(raw_artifact, pathlib.Path):
        if not raw_artifact.exists():
            raise MlflowException(f"{exception_header} with path '{raw_artifact}' does not exist.")
        if not raw_artifact.is_file():
            raise MlflowException(f"{exception_header} with path '{raw_artifact}' is not a file.")
        if raw_artifact.suffix not in _EXT_TO_ARTIFACT_MAP:
            raise MlflowException(
                f"{exception_header} with path '{raw_artifact}' does not match any of the supported"
                f" file extensions: {', '.join(_EXT_TO_ARTIFACT_MAP.keys())}."
            )
        return _InferredArtifactProperties(
            from_path=True, type=_EXT_TO_ARTIFACT_MAP[raw_artifact.suffix], ext=raw_artifact.suffix
        )

    # Type inference based on object type
    if type(raw_artifact) in _TYPE_TO_ARTIFACT_MAP:
        return _InferredArtifactProperties(
            from_path=False,
            type=_TYPE_TO_ARTIFACT_MAP[type(raw_artifact)],
            ext=_TYPE_TO_EXT_MAP[type(raw_artifact)],
        )

    # Given as other python object, we first attempt to infer as JsonEvaluationArtifact. If that
    # fails, we store it as PickleEvaluationArtifact
    try:
        json.dumps(raw_artifact, cls=NumpyEncoder)
        return _InferredArtifactProperties(
            from_path=False, type=JsonEvaluationArtifact, ext=".json"
        )
    except TypeError:
        return _InferredArtifactProperties(
            from_path=False, type=PickleEvaluationArtifact, ext=".pickle"
        )

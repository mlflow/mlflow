import pandas as pd
import numpy as np
import json

from mlflow.models.evaluation.base import EvaluationArtifact


class ImageEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.save(output_artifact_path)

    def _load_content_from_file(self, local_artifact_path):
        from PIL.Image import open as open_image

        self._content = open_image(local_artifact_path)
        return self._content


class CsvEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content


class ParquetEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        self._content.to_parquet(output_artifact_path, compression="brotli")

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_parquet(local_artifact_path)
        return self._content


class NpyEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        np.save(output_artifact_path, self._content, allow_pickle=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = np.load(local_artifact_path, allow_pickle=False)
        return self._content


class JsonEvaluationArtifact(EvaluationArtifact):
    def _save(self, output_artifact_path):
        json.dump(self._content, open(output_artifact_path, "w"))

    def _load_content_from_file(self, local_artifact_path):
        self._content = json.load(open(local_artifact_path, "r"))
        return self._content

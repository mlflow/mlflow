import pandas as pd

from mlflow.models.evaluation.base import EvaluationArtifact


class ImageEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        self._content.save(output_artifact_path)

    def _load_content_from_file(self, local_artifact_path):
        from PIL.Image import open as open_image

        self._content = open_image(local_artifact_path)
        return self._content


class CsvEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content

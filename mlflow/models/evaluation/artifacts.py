import pandas as pd

from mlflow.models.evaluation.base import EvaluationArtifact


class BinaryFileEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        with open(output_artifact_path, 'wb') as f:
            f.write(self._content)

    def _load_content_from_file(self, local_artifact_path):
        with open(local_artifact_path, 'rb') as f:
            self._content = f.read(-1)
        return self._content


class CsvEvaluationArtifact(EvaluationArtifact):
    def save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content

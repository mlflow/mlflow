import os
import requests

from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path


class HttpArtifactRepository(ArtifactRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)

        file_name = os.path.basename(local_file)
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        with open(local_file, "rb") as f:
            request_url = os.path.join(self.artifact_uri, *paths)
            resp = requests.post(request_url, data=f)
            resp.raise_for_status()

    def _is_directory(self, artifact_path):
        pass

    def log_artifacts(self, local_dir, artifact_path=None):
        pass

    def download_artifacts(self, artifact_path, dst_path=None):
        pass

    def list_artifacts(self, path=None):
        pass

    def _download_file(self, remote_file_path, local_path):
        pass

    def delete_artifacts(self, artifact_path=None):
        pass

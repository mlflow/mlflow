import os
import requests
import posixpath

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import augmented_raise_for_status


class HttpArtifactRepository(ArtifactRepository):
    """Stores artifacts in a remote artifact storage using HTTP requests"""

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)
        self._session = requests.Session()

    def __del__(self):
        if hasattr(self, "_session"):
            self._session.close()

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)

        file_name = os.path.basename(local_file)
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        url = posixpath.join(self.artifact_uri, *paths)
        with open(local_file, "rb") as f:
            resp = self._session.put(url, data=f, timeout=600)
            augmented_raise_for_status(resp)

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            if root == local_dir:
                artifact_dir = artifact_path
            else:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_dir = (
                    posixpath.join(artifact_path, rel_path) if artifact_path else rel_path
                )
            for f in filenames:
                self.log_artifact(os.path.join(root, f), artifact_dir)

    def list_artifacts(self, path=None):
        sep = "/mlflow-artifacts/artifacts"
        head, tail = self.artifact_uri.split(sep, maxsplit=1)
        url = head + sep
        root = tail.lstrip("/")
        params = {"path": posixpath.join(root, path) if path else root}
        resp = self._session.get(url, params=params, timeout=10)
        augmented_raise_for_status(resp)
        file_infos = []
        for f in resp.json().get("files", []):
            file_info = FileInfo(
                posixpath.join(path, f["path"]) if path else f["path"],
                f["is_dir"],
                int(f["file_size"]) if ("file_size" in f) else None,
            )
            file_infos.append(file_info)

        return sorted(file_infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        url = posixpath.join(self.artifact_uri, remote_file_path)
        with self._session.get(url, stream=True, timeout=10) as resp:
            augmented_raise_for_status(resp)
            with open(local_path, "wb") as f:
                chunk_size = 1024 * 1024  # 1 MB
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

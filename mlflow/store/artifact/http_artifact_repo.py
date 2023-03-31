import os
import posixpath

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request


class HttpArtifactRepository(ArtifactRepository):
    """Stores artifacts in a remote artifact storage using HTTP requests"""

    @property
    def _host_creds(self):
        return _get_default_host_creds(self.artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)

        file_name = os.path.basename(local_file)
        mime_type = _guess_mime_type(file_name)
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        endpoint = posixpath.join("/", *paths)
        extra_headers = {"Content-Type": mime_type}
        with open(local_file, "rb") as f:
            resp = http_request(
                self._host_creds, endpoint, "PUT", data=f, extra_headers=extra_headers
            )
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
        endpoint = "/mlflow-artifacts/artifacts"
        url, tail = self.artifact_uri.split(endpoint, maxsplit=1)
        root = tail.lstrip("/")
        params = {"path": posixpath.join(root, path) if path else root}
        host_creds = _get_default_host_creds(url)
        resp = http_request(host_creds, endpoint, "GET", params=params)
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
        endpoint = posixpath.join("/", remote_file_path)
        resp = http_request(self._host_creds, endpoint, "GET", stream=True)
        augmented_raise_for_status(resp)
        with open(local_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    def delete_artifacts(self, artifact_path=None):
        endpoint = posixpath.join("/", artifact_path) if artifact_path else "/"
        resp = http_request(self._host_creds, endpoint, "DELETE", stream=True)
        augmented_raise_for_status(resp)

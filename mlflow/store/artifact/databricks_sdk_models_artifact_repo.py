import posixpath

from mlflow.entities import FileInfo
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository

DOWNLOAD_CHUNK_SIZE = 1024 * 1024 * 1024


def _get_databricks_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


class DatabricksSDKModelsArtifactRepository(CloudArtifactRepository):
    """
    Stores and retrieves model artifacts via Databricks SDK, agnostic to the underlying cloud
    that stores the model artifacts.
    """

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.model_base_path = f"/Models/{model_name.replace('.', '/')}/{model_version}"
        self.client = _get_databricks_workspace_client()
        super().__init__(self.model_base_path)

    def list_artifacts(self, path=None):
        dest_path = self.model_base_path
        if path:
            dest_path = posixpath.join(dest_path, path)

        file_infos = []
        resp = self.client.files.list_directory_contents(dest_path)
        for directory_entry in resp:
            relative_path = posixpath.relpath(directory_entry.path, self.model_base_path)
            file_infos.append(
                FileInfo(
                    path=relative_path,
                    is_dir=directory_entry.is_directory,
                    file_size=directory_entry.file_size,
                )
            )

        return sorted(file_infos, key=lambda f: f.path)

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path=None):
        dest_path = self.model_base_path
        if artifact_file_path:
            dest_path = posixpath.join(dest_path, artifact_file_path)

        with open(src_file_path, "rb") as f:
            self.client.files.upload(dest_path, f, overwrite=True)

    def log_artifact(self, local_file, artifact_path=None):
        self._upload_to_cloud(
            cloud_credential_info=None,
            src_file_path=local_file,
            artifact_file_path=artifact_path,
        )

    def _download_from_cloud(self, remote_file_path, local_path):
        dest_path = self.model_base_path
        if remote_file_path:
            dest_path = posixpath.join(dest_path, remote_file_path)

        resp = self.client.files.download(dest_path)
        contents = resp.contents

        with open(local_path, "wb") as f:
            while chunk := contents.read(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

    def _get_write_credential_infos(self, remote_file_paths):
        # Databricks sdk based model download/upload don't need any extra credentials
        return [None] * len(remote_file_paths)

    def _get_read_credential_infos(self, remote_file_paths):
        # Databricks sdk based model download/upload don't need any extra credentials
        return [None] * len(remote_file_paths)

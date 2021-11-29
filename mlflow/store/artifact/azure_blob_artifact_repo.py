import os
import posixpath
import re
import urllib.parse

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository


class AzureBlobArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure Blob Storage.

    This repository is used with URIs of the form
    ``wasbs://<container-name>@<ystorage-account-name>.blob.core.windows.net/<path>``,
    following the same URI scheme as Hadoop on Azure blob storage. It requires either that:
    - Azure storage connection string is in the env var ``AZURE_STORAGE_CONNECTION_STRING``
    - Azure storage access key is in the env var ``AZURE_STORAGE_ACCESS_KEY``
    - DefaultAzureCredential is configured
    """

    def __init__(self, artifact_uri, client=None):
        super().__init__(artifact_uri)

        # Allow override for testing
        if client:
            self.client = client
            return

        from azure.storage.blob import BlobServiceClient

        (_, account, _, api_uri_suffix) = AzureBlobArtifactRepository.parse_wasbs_uri(artifact_uri)
        if "AZURE_STORAGE_CONNECTION_STRING" in os.environ:
            self.client = BlobServiceClient.from_connection_string(
                conn_str=os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            )
        elif "AZURE_STORAGE_ACCESS_KEY" in os.environ:
            account_url = "https://{account}.{api_uri_suffix}".format(
                account=account, api_uri_suffix=api_uri_suffix
            )
            self.client = BlobServiceClient(
                account_url=account_url, credential=os.environ.get("AZURE_STORAGE_ACCESS_KEY")
            )
        else:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ImportError(
                    "Using DefaultAzureCredential requires the azure-identity package. "
                    "Please install it via: pip install azure-identity"
                ) from exc

            account_url = "https://{account}.{api_uri_suffix}".format(
                account=account, api_uri_suffix=api_uri_suffix
            )
            self.client = BlobServiceClient(
                account_url=account_url, credential=DefaultAzureCredential()
            )

    @staticmethod
    def parse_wasbs_uri(uri):
        """Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix)."""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "wasbs":
            raise Exception("Not a WASBS URI: %s" % uri)

        match = re.match(
            r"([^@]+)@([^.]+)\.(blob\.core\.(windows\.net|chinacloudapi\.cn))", parsed.netloc
        )

        if match is None:
            raise Exception(
                "WASBS URI must be of the form "
                "<container>@<account>.blob.core.windows.net"
                " or <container>@<account>.blob.core.chinacloudapi.cn"
            )
        container = match.group(1)
        storage_account = match.group(2)
        api_uri_suffix = match.group(3)
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return container, storage_account, path, api_uri_suffix

    def log_artifact(self, local_file, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        with open(local_file, "rb") as file:
            container_client.upload_blob(dest_path, file, overwrite=True)

    def log_artifacts(self, local_dir, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                remote_file_path = posixpath.join(upload_path, f)
                local_file_path = os.path.join(root, f)
                with open(local_file_path, "rb") as file:
                    container_client.upload_blob(remote_file_path, file, overwrite=True)

    def list_artifacts(self, path=None):
        # Newer versions of `azure-storage-blob` (>= 12.4.0) provide a public
        # `azure.storage.blob.BlobPrefix` object to signify that a blob is a directory,
        # while older versions only expose this API internally as
        # `azure.storage.blob._models.BlobPrefix`
        try:
            from azure.storage.blob import BlobPrefix
        except ImportError:
            from azure.storage.blob._models import BlobPrefix

        (container, _, artifact_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"
        results = container_client.walk_blobs(name_starts_with=prefix)
        for r in results:
            if not r.name.startswith(artifact_path):
                raise MlflowException(
                    "The name of the listed Azure blob does not begin with the specified"
                    " artifact path. Artifact path: {artifact_path}. Blob name:"
                    " {blob_name}".format(artifact_path=artifact_path, blob_name=r.name)
                )
            if isinstance(r, BlobPrefix):  # This is a prefix for items in a subdirectory
                subdir = posixpath.relpath(path=r.name, start=artifact_path)
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, True, None))
            else:  # Just a plain old blob
                file_name = posixpath.relpath(path=r.name, start=artifact_path)
                infos.append(FileInfo(file_name, False, r.size))
        # The list_artifacts API expects us to return an empty list if the
        # the path references a single file.
        rel_path = dest_path[len(artifact_path) + 1 :]
        if (len(infos) == 1) and not infos[0].is_dir and (infos[0].path == rel_path):
            return []
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        (container, _, remote_root_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        with open(local_path, "wb") as file:
            container_client.download_blob(remote_full_path).readinto(file)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")

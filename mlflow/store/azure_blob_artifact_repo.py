import os
import re

from six.moves import urllib

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact_repo import ArtifactRepository


class AzureBlobArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure Blob Storage.

    This repository is used with URIs of the form
    ``wasbs://<container-name>@<ystorage-account-name>.blob.core.windows.net/<path>``,
    following the same URI scheme as Hadoop on Azure blob storage. It requires that your Azure
    storage access key be available in the environment variable ``AZURE_STORAGE_ACCESS_KEY``.
    """

    def __init__(self, artifact_uri, client=None):
        super(AzureBlobArtifactRepository, self).__init__(artifact_uri)

        # Allow override for testing
        if client:
            self.client = client
            return

        from azure.storage.blob import BlockBlobService
        (_, account, _) = AzureBlobArtifactRepository.parse_wasbs_uri(artifact_uri)
        if "AZURE_STORAGE_CONNECTION_STRING" in os.environ:
            self.client = BlockBlobService(
                account_name=account,
                connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))
        elif "AZURE_STORAGE_ACCESS_KEY" in os.environ:
            self.client = BlockBlobService(
                account_name=account,
                account_key=os.environ.get("AZURE_STORAGE_ACCESS_KEY"))
        else:
            raise Exception("You need to set one of AZURE_STORAGE_CONNECTION_STRING or "
                            "AZURE_STORAGE_ACCESS_KEY to access Azure storage.")

    @staticmethod
    def parse_wasbs_uri(uri):
        """Parse a wasbs:// URI, returning (container, storage_account, path)."""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "wasbs":
            raise Exception("Not a WASBS URI: %s" % uri)
        match = re.match("([^@]+)@([^.]+)\\.blob\\.core\\.windows\\.net", parsed.netloc)
        if match is None:
            raise Exception("WASBS URI must be of the form "
                            "<container>@<account>.blob.core.windows.net")
        container = match.group(1)
        storage_account = match.group(2)
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return container, storage_account, path

    def get_path_module(self):
        import posixpath
        return posixpath

    def log_artifact(self, local_file, artifact_path=None):
        (container, _, dest_path) = self.parse_wasbs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = self.get_path_module().join(dest_path, artifact_path)
        dest_path = self.get_path_module().join(
                dest_path, self.get_path_module().basename(local_file))
        self.client.create_blob_from_path(container, dest_path, local_file)

    def log_artifacts(self, local_dir, artifact_path=None):
        (container, _, dest_path) = self.parse_wasbs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = self.get_path_module().join(dest_path, artifact_path)
        local_dir = self.get_path_module().abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = self.get_path_module().relpath(root, local_dir)
                upload_path = self.get_path_module().join(dest_path, rel_path)
            for f in filenames:
                path = self.get_path_module().join(upload_path, f)
                self.client.create_blob_from_path(
                    container, path, self.get_path_module().join(root, f))

    def list_artifacts(self, path=None):
        from azure.storage.blob.models import BlobPrefix
        (container, _, artifact_path) = self.parse_wasbs_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = self.get_path_module().join(dest_path, path)
        infos = []
        prefix = dest_path + "/"
        marker = None  # Used to make next list request if this one exceeded the result limit
        while True:
            results = self.client.list_blobs(container, prefix=prefix, delimiter='/', marker=marker)
            for r in results:
                if not r.name.startswith(artifact_path):
                    raise MlflowException(
                        "The name of the listed Azure blob does not begin with the specified"
                        " artifact path. Artifact path: {artifact_path}. Blob name:"
                        " {blob_name}".format(artifact_path=artifact_path, blob_name=r.name))
                if isinstance(r, BlobPrefix):   # This is a prefix for items in a subdirectory
                    subdir = self.get_path_module().relpath(path=r.name, start=artifact_path)
                    if subdir.endswith("/"):
                        subdir = subdir[:-1]
                    infos.append(FileInfo(subdir, True, None))
                else:  # Just a plain old blob
                    file_name = self.get_path_module().relpath(path=r.name, start=artifact_path)
                    infos.append(FileInfo(file_name, False, r.properties.content_length))
            # Check whether a new marker is returned, meaning we have to make another request
            if results.next_marker:
                marker = results.next_marker
            else:
                break
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        (container, _, remote_root_path) = self.parse_wasbs_uri(self.artifact_uri)
        remote_full_path = self.get_path_module().join(remote_root_path, remote_file_path)
        self.client.get_blob_to_path(container, remote_full_path, local_path)

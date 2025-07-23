import base64
import datetime
import os
import posixpath
import re
import urllib.parse
from typing import Optional, Union

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
)
from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.credentials import get_default_host_creds


def encode_base64(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    encoded = base64.b64encode(data)
    return encoded.decode("utf-8")


def decode_base64(encoded: str) -> str:
    decoded_bytes = base64.b64decode(encoded)
    return decoded_bytes.decode("utf-8")


class AzureBlobArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """
    Stores artifacts on Azure Blob Storage.

    This repository is used with URIs of the form
    ``wasbs://<container-name>@<ystorage-account-name>.blob.core.windows.net/<path>``,
    following the same URI scheme as Hadoop on Azure blob storage. It requires either that:
    - Azure storage connection string is in the env var ``AZURE_STORAGE_CONNECTION_STRING``
    - Azure storage access key is in the env var ``AZURE_STORAGE_ACCESS_KEY``
    - DefaultAzureCredential is configured
    """

    def __init__(self, artifact_uri: str, client=None, tracking_uri: Optional[str] = None) -> None:
        super().__init__(artifact_uri, tracking_uri)

        _DEFAULT_TIMEOUT = 600  # 10 minutes
        self.write_timeout = MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get() or _DEFAULT_TIMEOUT

        # Allow override for testing
        if client:
            self.client = client
            return

        from azure.storage.blob import BlobServiceClient

        (_, account, _, api_uri_suffix) = AzureBlobArtifactRepository.parse_wasbs_uri(artifact_uri)
        if "AZURE_STORAGE_CONNECTION_STRING" in os.environ:
            self.client = BlobServiceClient.from_connection_string(
                conn_str=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
                connection_verify=get_default_host_creds(artifact_uri).verify,
            )
        elif "AZURE_STORAGE_ACCESS_KEY" in os.environ:
            account_url = f"https://{account}.{api_uri_suffix}"
            self.client = BlobServiceClient(
                account_url=account_url,
                credential=os.environ.get("AZURE_STORAGE_ACCESS_KEY"),
                connection_verify=get_default_host_creds(artifact_uri).verify,
            )
        else:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ImportError(
                    "Using DefaultAzureCredential requires the azure-identity package. "
                    "Please install it via: pip install azure-identity"
                ) from exc

            account_url = f"https://{account}.{api_uri_suffix}"
            self.client = BlobServiceClient(
                account_url=account_url,
                credential=DefaultAzureCredential(),
                connection_verify=get_default_host_creds(artifact_uri).verify,
            )

    @staticmethod
    def parse_wasbs_uri(uri):
        """Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix)."""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "wasbs":
            raise Exception(f"Not a WASBS URI: {uri}")

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
            container_client.upload_blob(
                dest_path, file, overwrite=True, timeout=self.write_timeout
            )

    def log_artifacts(self, local_dir, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                remote_file_path = posixpath.join(upload_path, f)
                local_file_path = os.path.join(root, f)
                with open(local_file_path, "rb") as file:
                    container_client.upload_blob(
                        remote_file_path, file, overwrite=True, timeout=self.write_timeout
                    )

    def list_artifacts(self, path=None):
        # Newer versions of `azure-storage-blob` (>= 12.4.0) provide a public
        # `azure.storage.blob.BlobPrefix` object to signify that a blob is a directory,
        # while older versions only expose this API internally as
        # `azure.storage.blob._models.BlobPrefix`
        try:
            from azure.storage.blob import BlobPrefix
        except ImportError:
            from azure.storage.blob._models import BlobPrefix

        def is_dir(result):
            return isinstance(result, BlobPrefix)

        (container, _, artifact_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"
        results = container_client.walk_blobs(name_starts_with=prefix)

        for result in results:
            if (
                dest_path == result.name
            ):  # result isn't actually a child of the path we're interested in, so skip it
                continue

            if not result.name.startswith(artifact_path):
                raise MlflowException(
                    "The name of the listed Azure blob does not begin with the specified"
                    f" artifact path. Artifact path: {artifact_path}. Blob name: {result.name}"
                )

            if is_dir(result):
                subdir = posixpath.relpath(path=result.name, start=artifact_path)
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, is_dir=True, file_size=None))
            else:  # Just a plain old blob
                file_name = posixpath.relpath(path=result.name, start=artifact_path)
                infos.append(FileInfo(file_name, is_dir=False, file_size=result.size))

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
        blob = container_client.download_blob(remote_full_path)
        with open(local_path, "wb") as file:
            blob.readinto(file)

    def delete_artifacts(self, artifact_path=None):
        from azure.core.exceptions import ResourceNotFoundError

        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        try:
            blobs = container_client.list_blobs(name_starts_with=dest_path)
            blob_list = list(blobs)
            if not blob_list:
                raise MlflowException(f"No such file or directory: '{dest_path}'")

            for blob in blob_list:
                container_client.delete_blob(blob.name)
        except ResourceNotFoundError:
            raise MlflowException(f"No such file or directory: '{dest_path}'")

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        # Put Block: https://learn.microsoft.com/en-us/rest/api/storageservices/put-block?tabs=microsoft-entra-id
        # SDK: https://learn.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#azure-storage-blob-blobclient-stage-block
        blob_url = posixpath.join(self.client.url, container, dest_path)
        sas_token = generate_blob_sas(
            account_name=self.client.account_name,
            container_name=container,
            blob_name=dest_path,
            account_key=self.client.credential.account_key,
            permission=BlobSasPermissions(read=True, write=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        )
        credentials = []
        for i in range(1, num_parts + 1):
            block_id = f"mlflow_block_{i}"
            # see https://github.com/Azure/azure-sdk-for-python/blob/18a66ef98c6f2153491489d3d7d2fe4a5849e4ac/sdk/storage/azure-storage-blob/azure/storage/blob/_blob_client.py#L2468
            safe_block_id = urllib.parse.quote(encode_base64(block_id), safe="")
            url = f"{blob_url}?comp=block&blockid={safe_block_id}&{sas_token}"
            credentials.append(
                MultipartUploadCredential(
                    url=url,
                    part_number=i,
                    headers={},
                )
            )
        return CreateMultipartUploadResponse(
            credentials=credentials,
            upload_id=None,
        )

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        block_ids = []
        for part in parts:
            qs = urllib.parse.urlparse(part.url).query
            block_id = urllib.parse.parse_qs(qs)["blockid"][0]
            block_id = decode_base64(urllib.parse.unquote(block_id))
            block_ids.append(block_id)
        blob_client = self.client.get_blob_client(container, dest_path)
        blob_client.commit_block_list(block_ids)

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        # There is no way to delete uncommitted blocks in Azure Blob Storage.
        # Instead, they are garbage collected within 7 days.
        # See https://docs.microsoft.com/en-us/rest/api/storageservices/put-block-list#remarks
        # The blob may already exist so we cannot delete it either.
        pass

import base64
import datetime
import os
import posixpath
import re
import urllib.parse
from datetime import timezone

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
)
from mlflow.entities.presigned_download import PresignedDownloadUrlResponse
from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
from mlflow.exceptions import (
    MlflowException,
    _UnsupportedMultipartUploadException,
    _UnsupportedPresignedDownloadException,
)
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    MultipartDownloadMixin,
    MultipartUploadMixin,
    _is_object_key_within_path,
)
from mlflow.store.artifact.s3_artifact_repo import _attachment_content_disposition
from mlflow.utils.credentials import get_default_host_creds


def encode_base64(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    encoded = base64.b64encode(data)
    return encoded.decode("utf-8")


def decode_base64(encoded: str) -> str:
    decoded_bytes = base64.b64decode(encoded)
    return decoded_bytes.decode("utf-8")


class AzureBlobArtifactRepository(ArtifactRepository, MultipartUploadMixin, MultipartDownloadMixin):
    """
    Stores artifacts on Azure Blob Storage.

    This repository is used with URIs of the form
    ``wasbs://<container-name>@<storage-account-name>.blob.core.windows.net/<path>``,
    following the same URI scheme as Hadoop on Azure blob storage. Also supports
    Azure China (``chinacloudapi.cn``) and Azure Government (``usgovcloudapi.net``)
    endpoints. It requires either that:
    - Azure storage connection string is in the env var ``AZURE_STORAGE_CONNECTION_STRING``
    - Azure storage access key is in the env var ``AZURE_STORAGE_ACCESS_KEY``
    - DefaultAzureCredential is configured
    """

    def __init__(
        self,
        artifact_uri: str,
        client=None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)

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
                    "Please install it via: pip install mlflow[azure]"
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
            raise MlflowException.invalid_parameter_value(f"Not a WASBS URI: {uri}")

        match = re.fullmatch(
            r"([^@]+)@([^.]+)\.(blob\.core\.(windows\.net|chinacloudapi\.cn|usgovcloudapi\.net))",
            parsed.netloc,
        )

        if match is None:
            raise MlflowException.invalid_parameter_value(
                "WASBS URI must be of the form "
                "<container>@<account>.blob.core.windows.net"
                " or <container>@<account>.blob.core.chinacloudapi.cn"
                " or <container>@<account>.blob.core.usgovcloudapi.net"
            )
        container = match.group(1)
        storage_account = match.group(2)
        api_uri_suffix = match.group(3)
        path = parsed.path
        path = path.removeprefix("/")
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
                subdir = subdir.removesuffix("/")
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
        from azure.core.exceptions import ResourceNotFoundError

        (container, _, remote_root_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        try:
            blob = container_client.download_blob(remote_full_path)
            with open(local_path, "wb") as file:
                blob.readinto(file)
        except ResourceNotFoundError as e:
            raise MlflowException(
                f"No such file or directory: '{remote_full_path}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e

    def delete_artifacts(self, artifact_path=None):
        from azure.core.exceptions import ResourceNotFoundError

        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        try:
            blobs = container_client.list_blobs(name_starts_with=dest_path)
            blob_list = [blob for blob in blobs if _is_object_key_within_path(blob.name, dest_path)]
            if not blob_list:
                raise MlflowException(f"No such file or directory: '{dest_path}'")

            for blob in blob_list:
                container_client.delete_blob(blob.name)
        except ResourceNotFoundError:
            raise MlflowException(f"No such file or directory: '{dest_path}'")

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        from azure.core.exceptions import HttpResponseError
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        # Put Block: https://learn.microsoft.com/en-us/rest/api/storageservices/put-block?tabs=microsoft-entra-id
        # SDK: https://learn.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#azure-storage-blob-blobclient-stage-block
        blob_url = posixpath.join(self.client.url, container, dest_path)
        now = datetime.datetime.now(timezone.utc)
        expiry = now + datetime.timedelta(hours=1)
        sas_kwargs = {
            "account_name": self.client.account_name,
            "container_name": container,
            "blob_name": dest_path,
            "permission": BlobSasPermissions(read=True, write=True),
            "expiry": expiry,
        }
        credential = self.client.credential
        if account_key := getattr(credential, "account_key", None):
            sas_kwargs["account_key"] = account_key
        elif hasattr(credential, "get_token"):
            start = now - datetime.timedelta(minutes=5)
            try:
                user_delegation_key = self.client.get_user_delegation_key(start, expiry)
            except HttpResponseError as e:
                if (
                    e.status_code == 403
                    and getattr(e, "error_code", None) == "AuthorizationPermissionMismatch"
                ):
                    raise _UnsupportedMultipartUploadException() from e
                raise
            sas_kwargs.update(user_delegation_key=user_delegation_key, start=start)
        else:
            raise _UnsupportedMultipartUploadException()

        sas_token = generate_blob_sas(
            **sas_kwargs,
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

    def _generate_sas_token(self, container, blob_name, permission, expiry, **kwargs):
        """Generate a SAS token for a blob, raising a NOT_IMPLEMENTED error if it cannot be signed.

        Uses the account key if the client has one, otherwise a short-lived user delegation key
        (Microsoft Entra ID credentials). Credentials that cannot mint a SAS token (SAS token or
        anonymous credentials, or an Entra ID identity that is not allowed to request a user
        delegation key) raise ``_UnsupportedPresignedDownloadException``.
        """
        from azure.core.exceptions import HttpResponseError
        from azure.storage.blob import generate_blob_sas

        now = datetime.datetime.now(timezone.utc)
        sas_kwargs = {
            "account_name": self.client.account_name,
            "container_name": container,
            "blob_name": blob_name,
            "permission": permission,
            "expiry": expiry,
            **kwargs,
        }
        credential = self.client.credential
        if account_key := getattr(credential, "account_key", None):
            sas_kwargs["account_key"] = account_key
        elif hasattr(credential, "get_token"):
            start = now - datetime.timedelta(minutes=5)
            try:
                user_delegation_key = self.client.get_user_delegation_key(start, expiry)
            except HttpResponseError as e:
                if (
                    e.status_code == 403
                    and getattr(e, "error_code", None) == "AuthorizationPermissionMismatch"
                ):
                    raise _UnsupportedPresignedDownloadException() from e
                raise
            sas_kwargs.update(user_delegation_key=user_delegation_key, start=start)
        else:
            raise _UnsupportedPresignedDownloadException()

        return generate_blob_sas(**sas_kwargs)

    def get_download_presigned_url(self, artifact_path, expiration=300):
        """Generate a presigned URL for downloading an artifact directly from Azure Blob Storage.

        Raises:
            MlflowException: ``RESOURCE_DOES_NOT_EXIST`` if the blob does not exist.
            _UnsupportedPresignedDownloadException: If the client's credentials cannot sign
                a SAS token (``NOT_IMPLEMENTED``, i.e. HTTP 501 from the server).
        """
        from azure.core.exceptions import ResourceNotFoundError
        from azure.storage.blob import BlobSasPermissions

        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        dest_path = posixpath.join(dest_path, artifact_path) if artifact_path else dest_path

        blob_client = self.client.get_blob_client(container, dest_path)
        try:
            properties = blob_client.get_blob_properties()
        except ResourceNotFoundError as e:
            raise MlflowException(
                f"No such file or directory: '{dest_path}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from e

        expiry = datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=expiration)
        # Serve the blob as a download with the artifact's own filename, same
        # rationale as S3ArtifactRepository.get_download_presigned_url.
        sas_token = self._generate_sas_token(
            container,
            dest_path,
            BlobSasPermissions(read=True),
            expiry,
            content_disposition=_attachment_content_disposition(posixpath.basename(dest_path)),
        )
        return PresignedDownloadUrlResponse(
            url=f"{blob_client.url}?{sas_token}", headers={}, file_size=properties.size
        )

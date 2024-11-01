import datetime
import importlib.metadata
import os
import posixpath
import urllib.parse
from collections import namedtuple

from packaging.version import Version

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
)
from mlflow.environment_variables import (
    MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT,
    MLFLOW_GCS_DEFAULT_TIMEOUT,
    MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_GCS_UPLOAD_CHUNK_SIZE,
)
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    MultipartUploadMixin,
    _retry_with_new_creds,
)
from mlflow.utils.file_utils import relative_path_to_artifact_path

GCSMPUArguments = namedtuple("GCSMPUArguments", ["transport", "url", "headers", "content_type"])


class GCSArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """
    Stores artifacts on Google Cloud Storage.

    Args:
        artifact_uri: URI of GCS bucket
        client: Optional. The client to use for GCS operations; a default
            client object will be created if unspecified, using default
            credentials as described in https://google-cloud.readthedocs.io/en/latest/core/auth.html
    """

    def __init__(self, artifact_uri, client=None, credential_refresh_def=None):
        super().__init__(artifact_uri)
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import storage as gcs_storage
        from google.cloud.storage.constants import _DEFAULT_TIMEOUT

        self._GCS_DOWNLOAD_CHUNK_SIZE = MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE.get()
        self._GCS_UPLOAD_CHUNK_SIZE = MLFLOW_GCS_UPLOAD_CHUNK_SIZE.get()
        self._GCS_DEFAULT_TIMEOUT = (
            MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get()
            or MLFLOW_GCS_DEFAULT_TIMEOUT.get()
            or _DEFAULT_TIMEOUT
        )
        # Method to use for refresh
        self.credential_refresh_def = credential_refresh_def
        # If the user-supplied timeout environment variable value is -1,
        # use `None` for `self._GCS_DEFAULT_TIMEOUT`
        # to use indefinite timeout
        self._GCS_DEFAULT_TIMEOUT = (
            None if self._GCS_DEFAULT_TIMEOUT == -1 else self._GCS_DEFAULT_TIMEOUT
        )
        if client is not None:
            self.client = client
        else:
            try:
                self.client = gcs_storage.Client()
            except DefaultCredentialsError:
                self.client = gcs_storage.Client.create_anonymous_client()

    @staticmethod
    def parse_gcs_uri(uri):
        """Parse an GCS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "gs":
            raise Exception(f"Not a GCS URI: {uri}")
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return parsed.netloc, path

    def _get_bucket(self, bucket):
        return self.client.bucket(bucket)

    def _refresh_credentials(self):
        from google.cloud.storage import Client
        from google.oauth2.credentials import Credentials

        (bucket, _) = self.parse_gcs_uri(self.artifact_uri)
        if not self.credential_refresh_def:
            return self._get_bucket(bucket)
        new_token = self.credential_refresh_def()
        credentials = Credentials(new_token["oauth_token"])
        self.client = Client(project="mlflow", credentials=credentials)
        return self._get_bucket(bucket)

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path, chunk_size=self._GCS_UPLOAD_CHUNK_SIZE)
        blob.upload_from_filename(local_file, timeout=self._GCS_DEFAULT_TIMEOUT)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        local_dir = os.path.abspath(local_dir)

        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                gcs_bucket = self._get_bucket(bucket)
                path = posixpath.join(upload_path, f)
                # For large models, we need to speculatively retry a credential refresh
                # and throw if it still fails.  We cannot use the built-in refresh because UC
                # does not return a refresh token with the oauth token
                file_name = os.path.join(root, f)

                def try_func(gcs_bucket):
                    gcs_bucket.blob(
                        path, chunk_size=self._GCS_UPLOAD_CHUNK_SIZE
                    ).upload_from_filename(file_name, timeout=self._GCS_DEFAULT_TIMEOUT)

                _retry_with_new_creds(
                    try_func=try_func, creds_func=self._refresh_credentials, orig_creds=gcs_bucket
                )

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = self.parse_gcs_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"

        bkt = self._get_bucket(bucket)

        infos = self._list_folders(bkt, prefix, artifact_path)

        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        for result in results:
            # skip blobs matching current directory path as list_blobs api
            # returns subdirectories as well
            if result.name == prefix:
                continue
            blob_path = result.name[len(artifact_path) + 1 :]
            infos.append(FileInfo(blob_path, False, result.size))

        return sorted(infos, key=lambda f: f.path)

    def _list_folders(self, bkt, prefix, artifact_path):
        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        dir_paths = set()
        for page in results.pages:
            dir_paths.update(page.prefixes)

        return [FileInfo(path[len(artifact_path) + 1 : -1], True, None) for path in dir_paths]

    def _download_file(self, remote_file_path, local_path):
        (bucket, remote_root_path) = self.parse_gcs_uri(self.artifact_uri)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        gcs_bucket = self._get_bucket(bucket)
        gcs_bucket.blob(
            remote_full_path, chunk_size=self._GCS_DOWNLOAD_CHUNK_SIZE
        ).download_to_filename(local_path, timeout=self._GCS_DEFAULT_TIMEOUT)

    def delete_artifacts(self, artifact_path=None):
        (bucket_name, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        gcs_bucket = self._get_bucket(bucket_name)
        blobs = gcs_bucket.list_blobs(prefix=f"{dest_path}")
        for blob in blobs:
            blob.delete()

    @staticmethod
    def _validate_support_mpu():
        if Version(importlib.metadata.version("google-cloud-storage")) < Version(
            "2.12.0"
        ) or Version(importlib.metadata.version("google-resumable-media")) < Version("2.6.0"):
            raise _UnsupportedMultipartUploadException()

    @staticmethod
    def _gcs_mpu_arguments(filename: str, blob) -> GCSMPUArguments:
        """See :py:func:`google.cloud.storage.transfer_manager.upload_chunks_concurrently`"""
        from google.cloud.storage.transfer_manager import _headers_from_metadata

        bucket = blob.bucket
        client = blob.client
        transport = blob._get_transport(client)

        hostname = client._connection.get_api_base_url_for_mtls()
        url = f"{hostname}/{bucket.name}/{blob.name}"

        base_headers, object_metadata, content_type = blob._get_upload_arguments(
            client, None, filename=filename, command="tm.upload_sharded"
        )
        headers = {**base_headers, **_headers_from_metadata(object_metadata)}

        if blob.user_project is not None:
            headers["x-goog-user-project"] = blob.user_project

        if blob.kms_key_name is not None and "cryptoKeyVersions" not in blob.kms_key_name:
            headers["x-goog-encryption-kms-key-name"] = blob.kms_key_name

        return GCSMPUArguments(
            transport=transport, url=url, headers=headers, content_type=content_type
        )

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        self._validate_support_mpu()
        from google.resumable_media.requests import XMLMPUContainer

        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        args = self._gcs_mpu_arguments(local_file, blob)
        container = XMLMPUContainer(args.url, local_file, headers=args.headers)
        container.initiate(transport=args.transport, content_type=args.content_type)
        upload_id = container.upload_id

        credentials = []
        for i in range(1, num_parts + 1):  # part number must be in [1, 10000]
            signed_url = blob.generate_signed_url(
                method="PUT",
                version="v4",
                expiration=datetime.timedelta(minutes=60),
                query_parameters={
                    "partNumber": i,
                    "uploadId": upload_id,
                },
            )
            credentials.append(
                MultipartUploadCredential(
                    url=signed_url,
                    part_number=i,
                    headers={},
                )
            )
        return CreateMultipartUploadResponse(
            credentials=credentials,
            upload_id=upload_id,
        )

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        self._validate_support_mpu()
        from google.resumable_media.requests import XMLMPUContainer

        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        args = self._gcs_mpu_arguments(local_file, blob)
        container = XMLMPUContainer(args.url, local_file, headers=args.headers)
        container._upload_id = upload_id
        for part in parts:
            container.register_part(part.part_number, part.etag)

        container.finalize(transport=args.transport)

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        self._validate_support_mpu()
        from google.resumable_media.requests import XMLMPUContainer

        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        args = self._gcs_mpu_arguments(local_file, blob)
        container = XMLMPUContainer(args.url, local_file, headers=args.headers)
        container._upload_id = upload_id
        container.cancel(transport=args.transport)

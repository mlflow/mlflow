import json
import os
import posixpath
import urllib.parse
from datetime import datetime, timezone
from functools import lru_cache
from mimetypes import guess_type

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
)
from mlflow.environment_variables import (
    MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_S3_EXPECTED_BUCKET_OWNER,
    MLFLOW_S3_IGNORE_TLS,
    MLFLOW_S3_UPLOAD_EXTRA_ARGS,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
)
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    MultipartUploadMixin,
)
from mlflow.utils.file_utils import relative_path_to_artifact_path

_MAX_CACHE_SECONDS = 300

BOTO_TO_MLFLOW_ERROR = {
    "AccessDenied": PERMISSION_DENIED,
    "NoSuchBucket": RESOURCE_DOES_NOT_EXIST,
    "NoSuchKey": RESOURCE_DOES_NOT_EXIST,
    "InvalidAccessKeyId": UNAUTHENTICATED,
    "SignatureDoesNotMatch": UNAUTHENTICATED,
}


def _get_utcnow_timestamp():
    return datetime.now(timezone.utc).timestamp()


@lru_cache(maxsize=64)
def _cached_get_s3_client(
    signature_version,
    addressing_style,
    s3_endpoint_url,
    verify,
    timestamp,
    access_key_id=None,
    secret_access_key=None,
    session_token=None,
    region_name=None,
):
    """Returns a boto3 client, caching to avoid extra boto3 verify calls.

    This method is outside of the S3ArtifactRepository as it is
    agnostic and could be used by other instances.

    `maxsize` set to avoid excessive memory consumption in the case
    a user has dynamic endpoints (intentionally or as a bug).

    Some of the boto3 endpoint urls, in very edge cases, might expire
    after twelve hours as that is the current expiration time. To ensure
    we throw an error on verification instead of using an expired endpoint
    we utilise the `timestamp` parameter to invalidate cache.
    """
    import boto3
    from botocore.client import Config

    # Making it possible to access public S3 buckets
    # Workaround for https://github.com/boto/botocore/issues/2442
    if signature_version.lower() == "unsigned":
        from botocore import UNSIGNED

        signature_version = UNSIGNED

    return boto3.client(
        "s3",
        config=Config(
            signature_version=signature_version, s3={"addressing_style": addressing_style}
        ),
        endpoint_url=s3_endpoint_url,
        verify=verify,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
        region_name=region_name,
    )


def _get_s3_client(
    addressing_style=None,
    access_key_id=None,
    secret_access_key=None,
    session_token=None,
    region_name=None,
    s3_endpoint_url=None,
):
    if not s3_endpoint_url:
        s3_endpoint_url = MLFLOW_S3_ENDPOINT_URL.get()
    do_verify = not MLFLOW_S3_IGNORE_TLS.get()

    # The valid verify argument value is None/False/path to cert bundle file, See
    # https://github.com/boto/boto3/blob/73865126cad3938ca80a2f567a1c79cb248169a7/
    # boto3/session.py#L212
    verify = None if do_verify else False

    # NOTE: If you need to specify this env variable, please file an issue at
    # https://github.com/mlflow/mlflow/issues so we know your use-case!
    signature_version = os.environ.get("MLFLOW_EXPERIMENTAL_S3_SIGNATURE_VERSION", "s3v4")

    # Invalidate cache every `_MAX_CACHE_SECONDS`
    timestamp = int(_get_utcnow_timestamp() / _MAX_CACHE_SECONDS)

    if not addressing_style:
        addressing_style = MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE.get()

    return _cached_get_s3_client(
        signature_version,
        addressing_style,
        s3_endpoint_url,
        verify,
        timestamp,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        region_name=region_name,
    )


class S3ArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """
    Stores artifacts on Amazon S3.

    This repository provides MLflow artifact storage using Amazon S3 as the backend.
    It supports both single-file uploads and multipart uploads for large files,
    with automatic content type detection and configurable upload parameters.

    The repository uses boto3 for S3 operations and supports various authentication
    methods including AWS credentials, IAM roles, and environment variables.

    Environment Variables:
        AWS_ACCESS_KEY_ID: AWS access key ID for authentication
        AWS_SECRET_ACCESS_KEY: AWS secret access key for authentication
        AWS_SESSION_TOKEN: AWS session token for temporary credentials
        AWS_DEFAULT_REGION: Default AWS region for S3 operations
        MLFLOW_S3_ENDPOINT_URL: Custom S3 endpoint URL (for S3-compatible storage)
        MLFLOW_S3_IGNORE_TLS: Set to 'true' to disable TLS verification
        MLFLOW_S3_UPLOAD_EXTRA_ARGS: JSON string of extra arguments for S3 uploads
        MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE: S3 addressing style ('path' or 'virtual')

    Note:
        This class inherits from both ArtifactRepository and MultipartUploadMixin,
        providing full artifact management capabilities including efficient large file uploads.
    """

    def __init__(
        self,
        artifact_uri: str,
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """
        Initialize an S3 artifact repository.

        Args:
            artifact_uri: S3 URI in the format 's3://bucket-name/path/to/artifacts'.
                The URI must be a valid S3 URI with a bucket that exists and is accessible.
            access_key_id: Optional AWS access key ID. If None, uses default AWS credential
                resolution (environment variables, IAM roles, etc.).
            secret_access_key: Optional AWS secret access key. Must be provided if
                access_key_id is provided.
            session_token: Optional AWS session token for temporary credentials.
                Used with STS tokens or IAM roles.
            tracking_uri: Optional URI for the MLflow tracking server.
                If None, uses the current tracking URI context.
        """
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._bucket_owner_params = (
            {"ExpectedBucketOwner": owner}
            if (owner := MLFLOW_S3_EXPECTED_BUCKET_OWNER.get())
            else {}
        )

    def _get_s3_client(self):
        return _get_s3_client(
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
        )

    def parse_s3_compliant_uri(self, uri):
        """
        Parse an S3 URI into bucket and path components.

        Args:
            uri: S3 URI in the format 's3://bucket-name/path/to/object'

        Returns:
            A tuple containing (bucket_name, object_path) where:
            - bucket_name: The S3 bucket name
            - object_path: The path within the bucket (without leading slash)
        """
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            raise Exception(f"Not an S3 URI: {uri}")
        path = parsed.path
        path = path.removeprefix("/")
        return parsed.netloc, path

    @staticmethod
    def get_s3_file_upload_extra_args():
        """
        Get additional S3 upload arguments from environment variables.

        Returns:
            Dictionary of extra arguments for S3 uploads, or None if not configured.
            These arguments are passed to boto3's upload_file method.

        Environment Variables:
            MLFLOW_S3_UPLOAD_EXTRA_ARGS: JSON string containing extra arguments
                for S3 uploads (e.g., '{"ServerSideEncryption": "AES256"}')
        """
        if s3_file_upload_extra_args := MLFLOW_S3_UPLOAD_EXTRA_ARGS.get():
            return json.loads(s3_file_upload_extra_args)
        else:
            return None

    def _upload_file(self, s3_client, local_file, bucket, key):
        extra_args = {}
        guessed_type, guessed_encoding = guess_type(local_file)
        if guessed_type is not None:
            extra_args["ContentType"] = guessed_type
        if guessed_encoding is not None:
            extra_args["ContentEncoding"] = guessed_encoding
        extra_args.update(self._bucket_owner_params)
        environ_extra_args = self.get_s3_file_upload_extra_args()
        if environ_extra_args is not None:
            extra_args.update(environ_extra_args)
        s3_client.upload_file(Filename=local_file, Bucket=bucket, Key=key, ExtraArgs=extra_args)

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact to S3.

        This method uploads a single file to S3 with automatic content type detection
        and optional extra upload arguments from environment variables.

        Args:
            local_file: Absolute path to the local file to upload. The file must
                exist and be readable.
            artifact_path: Optional relative path within the S3 bucket where the
                artifact should be stored. If None, the file is stored in the root
                of the configured S3 path. Use forward slashes (/) for path separators.
        """
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        self._upload_file(
            s3_client=self._get_s3_client(), local_file=local_file, bucket=bucket, key=dest_path
        )

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log all files in a local directory as artifacts to S3.

        This method recursively uploads all files in the specified directory,
        preserving the directory structure in S3. Each file is uploaded with
        automatic content type detection.

        Args:
            local_dir: Absolute path to the local directory containing files to upload.
                The directory must exist and be readable.
            artifact_path: Optional relative path within the S3 bucket where the
                artifacts should be stored. If None, files are stored in the root
                of the configured S3 path. Use forward slashes (/) for path separators.
        """
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        s3_client = self._get_s3_client()
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)

            for f in filenames:
                self._upload_file(
                    s3_client=s3_client,
                    local_file=os.path.join(root, f),
                    bucket=bucket,
                    key=posixpath.join(upload_path, f),
                )

    def _iterate_s3_paginated_results(self, bucket, prefix):
        """
        Iterate over paginated S3 list_objects_v2 results with error handling.

        This helper method isolates the S3 client operations that can raise ClientError
        and provides appropriate error handling and mapping to MLflow exceptions.
        The ClientError can occur during both paginator setup and iteration, because
        the botocore library makes lazy calls.

        Args:
            bucket: S3 bucket name
            prefix: S3 prefix to list objects under

        Yields:
            Individual result pages from S3 list_objects_v2 operation

        Raises:
            MlflowException: If S3 client operations fail
        """
        from botocore.exceptions import ClientError

        try:
            s3_client = self._get_s3_client()
            paginator = s3_client.get_paginator("list_objects_v2")
            results = paginator.paginate(
                Bucket=bucket, Prefix=prefix, Delimiter="/", **self._bucket_owner_params
            )
            for result in results:
                yield result
        except ClientError as error:
            error_code = error.response["Error"]["Code"]
            mlflow_error_code = BOTO_TO_MLFLOW_ERROR.get(error_code, INTERNAL_ERROR)
            error_message = error.response["Error"]["Message"]
            raise MlflowException(
                f"Failed to list artifacts in {self.artifact_uri}: {error_message}",
                error_code=mlflow_error_code,
            )

    def list_artifacts(self, path=None):
        """
        List all artifacts directly under the specified S3 path.

        This method uses S3's list_objects_v2 API with pagination to efficiently
        list artifacts. It treats S3 prefixes as directories and returns both
        files and directories as FileInfo objects.

        Args:
            path: Optional relative path within the S3 bucket to list. If None,
                lists artifacts in the root of the configured S3 path. If the path
                refers to a single file, returns an empty list per MLflow convention.

        Returns:
            A list of FileInfo objects representing artifacts directly under the
            specified path. Each FileInfo contains:
            - path: Relative path of the artifact from the repository root
            - is_dir: True if the artifact represents a directory (S3 prefix)
            - file_size: Size in bytes for files, None for directories
        """
        (bucket, artifact_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        dest_path = dest_path.rstrip("/") if dest_path else ""
        infos = []
        prefix = dest_path + "/" if dest_path else ""
        for result in self._iterate_s3_paginated_results(bucket, prefix):
            # Subdirectories will be listed as "common prefixes"
            # due to the way we made the request
            for obj in result.get("CommonPrefixes", []):
                subdir_path = obj.get("Prefix")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=subdir_path, artifact_path=artifact_path
                )
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                subdir_rel_path = subdir_rel_path.removesuffix("/")
                infos.append(FileInfo(subdir_rel_path, True, None))
            # Objects listed directly will be files
            for obj in result.get("Contents", []):
                file_path = obj.get("Key")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=file_path, artifact_path=artifact_path
                )
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = int(obj.get("Size"))
                infos.append(FileInfo(file_rel_path, False, file_size))
        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed S3 object does not begin with the specified"
                f" artifact path. Artifact path: {artifact_path}. Object path:"
                f" {listed_object_path}."
            )

    def _download_file(self, remote_file_path, local_path):
        """
        Download a file from S3 to the local filesystem.

        This method downloads a single file from S3 to the specified local path.
        It's used internally by the download_artifacts method.

        Args:
            remote_file_path: Relative path of the file within the S3 bucket,
                relative to the repository's root path.
            local_path: Absolute path where the file should be saved locally.
                The parent directory must exist.
        """
        (bucket, s3_root_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        s3_full_path = posixpath.join(s3_root_path, remote_file_path)
        s3_client = self._get_s3_client()
        download_kwargs = (
            {"ExtraArgs": self._bucket_owner_params} if self._bucket_owner_params else {}
        )
        s3_client.download_file(bucket, s3_full_path, local_path, **download_kwargs)

    def delete_artifacts(self, artifact_path=None):
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        dest_path = dest_path.rstrip("/") if dest_path else ""
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        results = paginator.paginate(Bucket=bucket, Prefix=dest_path, **self._bucket_owner_params)
        for result in results:
            keys = []
            for to_delete_obj in result.get("Contents", []):
                file_path = to_delete_obj.get("Key")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=file_path, artifact_path=dest_path
                )
                keys.append({"Key": file_path})
            if keys:
                s3_client.delete_objects(
                    Bucket=bucket, Delete={"Objects": keys}, **self._bucket_owner_params
                )

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        """
        Initiate a multipart upload for efficient large file uploads to S3.

        This method creates a multipart upload session in S3 and generates
        presigned URLs for uploading each part. This is more efficient than
        single-part uploads for large files and provides better error recovery.

        Args:
            local_file: Absolute path to the local file to upload. The file must
                exist and be readable.
            num_parts: Number of parts to split the upload into. Must be between
                1 and 10,000 (S3 limit). More parts allow greater parallelism
                but increase overhead.
            artifact_path: Optional relative path within the S3 bucket where the
                artifact should be stored. If None, the file is stored in the root
                of the configured S3 path.

        Returns:
            CreateMultipartUploadResponse containing:
            - credentials: List of MultipartUploadCredential objects with presigned URLs
            - upload_id: S3 upload ID for tracking this multipart upload
        """
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        s3_client = self._get_s3_client()
        create_response = s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=dest_path,
            **self._bucket_owner_params,
        )
        upload_id = create_response["UploadId"]
        credentials = []
        for i in range(1, num_parts + 1):  # part number must be in [1, 10000]
            url = s3_client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": dest_path,
                    "PartNumber": i,
                    "UploadId": upload_id,
                    **self._bucket_owner_params,
                },
            )
            credentials.append(
                MultipartUploadCredential(
                    url=url,
                    part_number=i,
                    headers={},
                )
            )
        return CreateMultipartUploadResponse(
            credentials=credentials,
            upload_id=upload_id,
        )

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        """
        Complete a multipart upload by combining all parts into a single S3 object.

        This method should be called after all parts have been successfully uploaded
        using the presigned URLs from create_multipart_upload. It tells S3 to combine
        all the parts into the final object.

        Args:
            local_file: Absolute path to the local file that was uploaded. Must match
                the local_file used in create_multipart_upload.
            upload_id: The S3 upload ID returned by create_multipart_upload.
            parts: List of MultipartUploadPart objects containing metadata for each
                successfully uploaded part. Must include part_number and etag for each part.
                Parts must be provided in order (part 1, part 2, etc.).
            artifact_path: Optional relative path where the artifact should be stored.
                Must match the artifact_path used in create_multipart_upload.
        """
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        parts = [{"PartNumber": part.part_number, "ETag": part.etag} for part in parts]
        s3_client = self._get_s3_client()
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=dest_path,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
            **self._bucket_owner_params,
        )

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        """
        Abort a multipart upload and clean up any uploaded parts.

        This method should be called if a multipart upload fails or is cancelled.
        It cleans up any parts that were successfully uploaded and cancels the
        multipart upload session in S3.

        Args:
            local_file: Absolute path to the local file that was being uploaded.
                Must match the local_file used in create_multipart_upload.
            upload_id: The S3 upload ID returned by create_multipart_upload.
            artifact_path: Optional relative path where the artifact would have been stored.
                Must match the artifact_path used in create_multipart_upload.
        """
        (bucket, dest_path) = self.parse_s3_compliant_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        s3_client = self._get_s3_client()
        s3_client.abort_multipart_upload(
            Bucket=bucket,
            Key=dest_path,
            UploadId=upload_id,
            **self._bucket_owner_params,
        )

import json
import logging
import math
import os
import posixpath
import urllib.parse
from datetime import datetime
from functools import lru_cache
from mimetypes import guess_type

import requests

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_ENABLE_MULTIPART_UPLOAD,
    MLFLOW_S3_ENDPOINT_URL,
    MLFLOW_S3_IGNORE_TLS,
    MLFLOW_S3_UPLOAD_EXTRA_ARGS,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
    _MULTIPART_UPLOAD_CHUNK_SIZE,
    CloudArtifactRepository,
    _complete_futures,
)
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status

_logger = logging.getLogger(__name__)

_MAX_CACHE_SECONDS = 300


def _get_utcnow_timestamp():
    return datetime.utcnow().timestamp()


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
):  # pylint: disable=unused-argument
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
    addressing_style="path",
    access_key_id=None,
    secret_access_key=None,
    session_token=None,
    region_name=None,
):
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


class S3ArtifactRepository(CloudArtifactRepository):
    """Stores artifacts on Amazon S3."""

    def __init__(
        self,
        artifact_uri,
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        addressing_style="path",
    ):
        super().__init__(artifact_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._addressing_style = addressing_style
        self.bucket, self.bucket_path = self.parse_s3_compliant_uri(self.artifact_uri)
        self._set_region_name()

    def _set_region_name(self):
        temp_client = _get_s3_client(
            addressing_style=self._addressing_style,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
        )
        self.region_name = temp_client.get_bucket_location(Bucket=self.bucket)["LocationConstraint"]

    def _get_s3_client(self):
        return _get_s3_client(
            addressing_style=self._addressing_style,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            region_name=self.region_name,
        )

    def parse_s3_compliant_uri(self, uri):
        """Parse an S3 URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            raise Exception(f"Not an S3 URI: {uri}")
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return parsed.netloc, path

    @staticmethod
    def get_s3_file_upload_extra_args():
        s3_file_upload_extra_args = MLFLOW_S3_UPLOAD_EXTRA_ARGS.get()
        if s3_file_upload_extra_args:
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
        environ_extra_args = self.get_s3_file_upload_extra_args()
        if environ_extra_args is not None:
            extra_args.update(environ_extra_args)
        s3_client.upload_file(Filename=local_file, Bucket=bucket, Key=key, ExtraArgs=extra_args)

    def log_artifact(self, local_file, artifact_path=None):
        dest_path = self.bucket_path
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        key = posixpath.normpath(dest_path)
        self._upload_file(
            s3_client=self._get_s3_client(),
            local_file=local_file,
            bucket=self.bucket,
            key=key,
        )

    def _get_write_credential_infos(self, remote_file_paths):
        """
        Instead of returning ArtifactCredentialInfo objects, we instead return a list of initialized
        S3 client. We do so because S3 clients cannot be instantiated within each thread.
        """
        return [self._get_s3_client() for _ in remote_file_paths]

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        dest_path = posixpath.join(self.bucket_path, artifact_file_path)
        key = posixpath.normpath(dest_path)
        if (
            MLFLOW_ENABLE_MULTIPART_UPLOAD.get()
            and os.path.getsize(src_file_path) > _MULTIPART_UPLOAD_CHUNK_SIZE
        ):
            self._multipart_upload(cloud_credential_info, src_file_path, self.bucket, key)
        else:
            self._upload_file(cloud_credential_info, src_file_path, self.bucket, key)

    def _multipart_upload(self, cloud_credential_info, local_file, bucket, key):
        # Create multipart upload
        s3_client = cloud_credential_info
        response = s3_client.create_multipart_upload(Bucket=bucket, Key=key)
        upload_id = response["UploadId"]

        # Create presigned URL for each part
        num_parts = math.ceil(os.path.getsize(local_file) / _MULTIPART_UPLOAD_CHUNK_SIZE)

        def _get_presigned_upload_part_url(part_number):
            return s3_client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
            )

        presigned_urls = [
            _get_presigned_upload_part_url(part_number) for part_number in range(1, num_parts + 1)
        ]

        # define helper functions for uploading data
        def _upload_part(presigned_url, local_file, start_byte, size):
            data = read_chunk(local_file, size, start_byte)
            with cloud_storage_http_request("put", presigned_url, data=data) as response:
                augmented_raise_for_status(response)
                return response.headers["ETag"]

        try:
            # Upload each part with retries
            futures = {}
            for index, presigned_url in enumerate(presigned_urls):
                part_number = index + 1
                start_byte = index * _MULTIPART_UPLOAD_CHUNK_SIZE
                future = self.chunk_thread_pool.submit(
                    _upload_part,
                    presigned_url=presigned_url,
                    local_file=local_file,
                    start_byte=start_byte,
                    size=_MULTIPART_UPLOAD_CHUNK_SIZE,
                )
                futures[future] = part_number

            results, errors = _complete_futures(futures, local_file)
            if errors:
                raise MlflowException(
                    f"Failed to upload at least one part of {local_file}. Errors: {errors}"
                )
            parts = [
                {"PartNumber": part_number, "ETag": results[part_number]}
                for part_number in sorted(results)
            ]

            # Complete multipart upload
            s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception as e:
            _logger.warning(
                "Encountered an unexpected error during multipart upload: %s, aborting", e
            )
            s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise e

    def list_artifacts(self, path=None):
        artifact_path = self.bucket_path
        dest_path = self.bucket_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path + "/" if dest_path else ""
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        results = paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
        for result in results:
            # Subdirectories will be listed as "common prefixes" due to the way we made the request
            for obj in result.get("CommonPrefixes", []):
                subdir_path = obj.get("Prefix")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=subdir_path, artifact_path=artifact_path
                )
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                if subdir_rel_path.endswith("/"):
                    subdir_rel_path = subdir_rel_path[:-1]
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

    def _get_presigned_uri(self, remote_file_path):
        s3_client = self._get_s3_client()
        s3_full_path = posixpath.join(self.bucket_path, remote_file_path)
        return s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": s3_full_path}
        )

    def _get_read_credential_infos(self, remote_file_paths):
        return [
            ArtifactCredentialInfo(signed_uri=self._get_presigned_uri(path))
            for path in remote_file_paths
        ]

    def _download_from_cloud(self, remote_file_path, local_path):
        s3_client = self._get_s3_client()
        s3_full_path = posixpath.join(self.bucket_path, remote_file_path)
        s3_client.download_file(self.bucket, s3_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        dest_path = self.bucket_path
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        s3_client = self._get_s3_client()
        list_objects = s3_client.list_objects(Bucket=self.bucket, Prefix=dest_path).get(
            "Contents", []
        )
        for to_delete_obj in list_objects:
            file_path = to_delete_obj.get("Key")
            self._verify_listed_object_contains_artifact_path_prefix(
                listed_object_path=file_path, artifact_path=dest_path
            )
            s3_client.delete_object(Bucket=self.bucket, Key=file_path)

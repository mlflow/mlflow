import json
import logging
import os
import posixpath
import urllib.parse
from mimetypes import guess_type

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_ENABLE_MULTIPART_UPLOAD,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_S3_EXPECTED_BUCKET_OWNER,
    MLFLOW_S3_UPLOAD_EXTRA_ARGS,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.artifact_repo import _retry_with_new_creds
from mlflow.store.artifact.cloud_artifact_repo import (
    CloudArtifactRepository,
    _complete_futures,
    _compute_num_chunks,
    _validate_chunk_size_aws,
)
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status

_logger = logging.getLogger(__name__)
_BUCKET_REGION = "BucketRegion"
_RESPONSE_METADATA = "ResponseMetadata"
_HTTP_HEADERS = "HTTPHeaders"
_HTTP_HEADER_BUCKET_REGION = "x-amz-bucket-region"
_BUCKET_LOCATION_NAME = "BucketLocationName"


class OptimizedS3ArtifactRepository(CloudArtifactRepository):
    """
    An optimized version of the S3 Artifact Repository.

    This class is used for uploading and downloading S3 artifacts for UC models. While it largely
    copies the behavior of the S3ArtifactRepository, the `log_artifact`, `log_artifacts`, and
    `_download_file` methods are optimized by replacing boto3 client operations with the use of
    presigned URLs for both uploads and downloads.
    """

    def __init__(
        self,
        artifact_uri,
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        credential_refresh_def=None,
        addressing_style=None,
        s3_endpoint_url=None,
        s3_upload_extra_args=None,
        tracking_uri=None,
        registry_uri: str | None = None,
    ):
        super().__init__(artifact_uri, tracking_uri=tracking_uri, registry_uri=registry_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._credential_refresh_def = credential_refresh_def
        self._addressing_style = addressing_style
        self._s3_endpoint_url = s3_endpoint_url
        self.bucket, self.bucket_path = self.parse_s3_compliant_uri(self.artifact_uri)
        self._bucket_owner_params = (
            {"ExpectedBucketOwner": owner}
            if (owner := MLFLOW_S3_EXPECTED_BUCKET_OWNER.get())
            else {}
        )
        self._region_name = self._get_region_name()
        self._s3_upload_extra_args = s3_upload_extra_args or {}

    def _refresh_credentials(self):
        if not self._credential_refresh_def:
            return self._get_s3_client()
        new_creds = self._credential_refresh_def()
        self._access_key_id = new_creds["access_key_id"]
        self._secret_access_key = new_creds["secret_access_key"]
        self._session_token = new_creds["session_token"]
        self._s3_upload_extra_args = new_creds["s3_upload_extra_args"]
        return self._get_s3_client()

    def _get_region_name(self):
        from botocore.exceptions import ClientError

        temp_client = _get_s3_client(
            addressing_style=self._addressing_style,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            s3_endpoint_url=self._s3_endpoint_url,
        )
        try:
            head_bucket_resp = temp_client.head_bucket(
                Bucket=self.bucket, **self._bucket_owner_params
            )
            # A normal response will have the region in the Bucket_Region field of the response
            if _BUCKET_REGION in head_bucket_resp:
                return head_bucket_resp[_BUCKET_REGION]
            # If the bucket exists but the caller does not have permissions, the http headers
            # are passed back as part of the metadata of a normal, non-throwing response.  In
            # this case we use the x-amz-bucket-region field of the HTTP headers which should
            # always be populated with the region.
            if (
                _RESPONSE_METADATA in head_bucket_resp
                and _HTTP_HEADERS in head_bucket_resp[_RESPONSE_METADATA]
                and _HTTP_HEADER_BUCKET_REGION
                in head_bucket_resp[_RESPONSE_METADATA][_HTTP_HEADERS]
            ):
                return head_bucket_resp[_RESPONSE_METADATA][_HTTP_HEADERS][
                    _HTTP_HEADER_BUCKET_REGION
                ]
            # Directory buckets do not have a Bucket_Region and instead have a
            # Bucket_Location_Name.  This name cannot be used as the region name
            # however, so we warn that this has happened and allow the exception
            # at the end to be raised.
            if _BUCKET_LOCATION_NAME in head_bucket_resp:
                _logger.warning(
                    f"Directory bucket {self.bucket} found with BucketLocationName "
                    f"{head_bucket_resp[_BUCKET_LOCATION_NAME]}."
                )
            raise Exception(f"Unable to get the region name for bucket {self.bucket}.")
        except ClientError as error:
            # If a client error occurs, we check to see if the x-amz-bucket-region field is set
            # in the response and return that.  If it is not present, this will raise due to the
            # key not being present.
            return error.response[_RESPONSE_METADATA][_HTTP_HEADERS][_HTTP_HEADER_BUCKET_REGION]

    def _get_s3_client(self):
        return _get_s3_client(
            addressing_style=self._addressing_style,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            region_name=self._region_name,
            s3_endpoint_url=self._s3_endpoint_url,
        )

    def parse_s3_compliant_uri(self, uri):
        """Parse an S3 URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            raise Exception(f"Not an S3 URI: {uri}")
        path = parsed.path
        path = path.removeprefix("/")
        return parsed.netloc, path

    @staticmethod
    def get_s3_file_upload_extra_args():
        if s3_file_upload_extra_args := MLFLOW_S3_UPLOAD_EXTRA_ARGS.get():
            return json.loads(s3_file_upload_extra_args)
        else:
            return None

    def _upload_file(self, s3_client, local_file, bucket, key):
        extra_args = {}
        extra_args.update(self._s3_upload_extra_args)
        guessed_type, guessed_encoding = guess_type(local_file)
        if guessed_type is not None:
            extra_args["ContentType"] = guessed_type
        if guessed_encoding is not None:
            extra_args["ContentEncoding"] = guessed_encoding
        extra_args.update(self._bucket_owner_params)
        environ_extra_args = self.get_s3_file_upload_extra_args()
        if environ_extra_args is not None:
            extra_args.update(environ_extra_args)

        def try_func(creds):
            creds.upload_file(Filename=local_file, Bucket=bucket, Key=key, ExtraArgs=extra_args)

        _retry_with_new_creds(
            try_func=try_func, creds_func=self._refresh_credentials, orig_creds=s3_client
        )

    def log_artifact(self, local_file, artifact_path=None):
        artifact_file_path = os.path.basename(local_file)
        if artifact_path:
            artifact_file_path = posixpath.join(artifact_path, artifact_file_path)
        self._upload_to_cloud(
            cloud_credential_info=self._get_s3_client(),
            src_file_path=local_file,
            artifact_file_path=artifact_file_path,
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
            and os.path.getsize(src_file_path) > MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
        ):
            self._multipart_upload(cloud_credential_info, src_file_path, self.bucket, key)
        else:
            self._upload_file(cloud_credential_info, src_file_path, self.bucket, key)

    def _multipart_upload(self, cloud_credential_info, local_file, bucket, key):
        # Create multipart upload
        s3_client = cloud_credential_info
        response = s3_client.create_multipart_upload(
            Bucket=bucket, Key=key, **self._bucket_owner_params
        )
        upload_id = response["UploadId"]

        num_parts = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
        _validate_chunk_size_aws(MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())

        # define helper functions for uploading data
        def _upload_part(part_number, local_file, start_byte, size):
            data = read_chunk(local_file, size, start_byte)

            def try_func(creds):
                # Create presigned URL for each part
                presigned_url = creds.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": bucket,
                        "Key": key,
                        "UploadId": upload_id,
                        "PartNumber": part_number,
                        **self._bucket_owner_params,
                    },
                )
                with cloud_storage_http_request("put", presigned_url, data=data) as response:
                    augmented_raise_for_status(response)
                    return response.headers["ETag"]

            return _retry_with_new_creds(
                try_func=try_func, creds_func=self._refresh_credentials, orig_creds=s3_client
            )

        try:
            # Upload each part with retries
            futures = {}
            for index in range(num_parts):
                part_number = index + 1
                start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                future = self.chunk_thread_pool.submit(
                    _upload_part,
                    part_number=part_number,
                    local_file=local_file,
                    start_byte=start_byte,
                    size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(),
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
                **self._bucket_owner_params,
            )
        except Exception as e:
            _logger.warning(
                "Encountered an unexpected error during multipart upload: %s, aborting", e
            )
            s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                **self._bucket_owner_params,
            )
            raise e

    def list_artifacts(self, path=None):
        artifact_path = self.bucket_path
        dest_path = self.bucket_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        dest_path = dest_path.rstrip("/") if dest_path else ""
        prefix = dest_path + "/" if dest_path else ""
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        results = paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
            **self._bucket_owner_params,
        )
        for result in results:
            # Subdirectories will be listed as "common prefixes" due to the way we made the request
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

    def _get_presigned_uri(self, remote_file_path):
        s3_client = self._get_s3_client()
        s3_full_path = posixpath.join(self.bucket_path, remote_file_path)
        return s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": s3_full_path,
                **self._bucket_owner_params,
            },
        )

    def _get_read_credential_infos(self, remote_file_paths):
        return [
            ArtifactCredentialInfo(signed_uri=self._get_presigned_uri(path))
            for path in remote_file_paths
        ]

    def _download_from_cloud(self, remote_file_path, local_path):
        s3_client = self._get_s3_client()
        s3_full_path = posixpath.join(self.bucket_path, remote_file_path)

        def try_func(creds):
            download_kwargs = (
                {"ExtraArgs": self._bucket_owner_params} if self._bucket_owner_params else {}
            )
            creds.download_file(self.bucket, s3_full_path, local_path, **download_kwargs)

        _retry_with_new_creds(
            try_func=try_func, creds_func=self._refresh_credentials, orig_creds=s3_client
        )

    def delete_artifacts(self, artifact_path=None):
        dest_path = self.bucket_path
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        dest_path = dest_path.rstrip("/") if dest_path else ""
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        results = paginator.paginate(
            Bucket=self.bucket,
            Prefix=dest_path,
            **self._bucket_owner_params,
        )
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
                    Bucket=self.bucket,
                    Delete={"Objects": keys},
                    **self._bucket_owner_params,
                )

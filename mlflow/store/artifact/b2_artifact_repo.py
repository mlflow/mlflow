from urllib.parse import urlparse

from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client

_B2_USER_AGENT = "b2ai-mlflow"


def _add_b2_user_agent(request, **kwargs):
    ua = request.headers.get("User-Agent", "")
    if _B2_USER_AGENT not in ua:
        request.headers["User-Agent"] = f"{ua} {_B2_USER_AGENT}"


class B2ArtifactRepository(OptimizedS3ArtifactRepository):
    """Stores artifacts on Backblaze B2."""

    def __init__(
        self,
        artifact_uri,
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        credential_refresh_def=None,
        s3_upload_extra_args=None,
        tracking_uri=None,
        registry_uri: str | None = None,
    ):
        s3_endpoint_url = self.convert_b2_uri_to_s3_endpoint_url(artifact_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._s3_endpoint_url = s3_endpoint_url
        self.bucket, self.bucket_path = self.parse_s3_compliant_uri(artifact_uri)
        super().__init__(
            artifact_uri,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=credential_refresh_def,
            addressing_style="path",
            s3_endpoint_url=s3_endpoint_url,
            s3_upload_extra_args=s3_upload_extra_args,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )

    @staticmethod
    def _register_b2_user_agent(client):
        client.meta.events.register("before-sign.s3", _add_b2_user_agent, unique_id="b2-user-agent")
        return client

    def _get_region_name(self):
        # Parse region from the endpoint URL (e.g. https://s3.us-west-004.backblazeb2.com)
        host = urlparse(self._s3_endpoint_url).hostname
        match host.split("."):
            case ["s3", region, "backblazeb2", "com"]:
                return region
            case _:
                raise Exception(f"Unable to parse region from B2 endpoint: {self._s3_endpoint_url}")

    def _get_s3_client(self):
        client = _get_s3_client(
            addressing_style="path",
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            region_name=self._region_name,
            s3_endpoint_url=self._s3_endpoint_url,
        )
        return self._register_b2_user_agent(client)

    def parse_s3_compliant_uri(self, uri):
        # b2 uri format: b2://<bucket-name>@<endpoint-host>/path
        parsed = urlparse(uri)
        if parsed.scheme != "b2":
            raise Exception(f"Not a B2 URI: {uri}")

        host = parsed.netloc
        path = parsed.path

        bucket = host.split("@")[0]
        path = path.removeprefix("/")
        return bucket, path

    @staticmethod
    def convert_b2_uri_to_s3_endpoint_url(b2_uri):
        host = urlparse(b2_uri).netloc
        host_without_bucket = host.split("@")[-1]
        return f"https://{host_without_bucket}"

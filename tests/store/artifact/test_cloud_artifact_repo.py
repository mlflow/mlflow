from concurrent.futures import Future
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo, ArtifactCredentialType
from mlflow.store.artifact.cloud_artifact_repo import (
    _ARTIFACT_UPLOAD_BATCH_SIZE,
    CloudArtifactRepository,
    FileUploadPlan,
    StagedArtifactUpload,
    _readable_size,
    _validate_chunk_size_aws,
)


@pytest.mark.parametrize(
    ("size", "size_str"), [(5 * 1024**2, "5.00 MB"), (712.345 * 1024**2, "712.35 MB")]
)
def test_readable_size(size, size_str):
    assert _readable_size(size) == size_str


def test_chunk_size_validation_failure():
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**2 - 1)
    with pytest.raises(MlflowException, match="Multipart chunk size"):
        _validate_chunk_size_aws(5 * 1024**3 + 1)


@pytest.mark.parametrize(
    ("future_result", "expected_call_count"),
    [
        (None, 2),  # Simulate where creds are expired, but successfully download after refresh
        (Exception("fake_exception"), 4),
        # Simulate where there is a download failure and retries are exhausted
    ],
)
def test__parallelized_download_from_cloud(
    monkeypatch, future_result, expected_call_count, tmp_path
):
    # Mock environment variables
    monkeypatch.setenv("_MLFLOW_MPD_NUM_RETRIES", "3")
    monkeypatch.setenv("_MLFLOW_MPD_RETRY_INTERVAL_SECONDS", "0")

    with mock.patch(
        "mlflow.store.artifact.cloud_artifact_repo.CloudArtifactRepository"
    ) as cloud_artifact_mock:
        cloud_artifact_instance = cloud_artifact_mock.return_value

        # Mock all methods except '_parallelized_download_from_cloud'
        cloud_artifact_instance._parallelized_download_from_cloud.side_effect = (
            lambda *args, **kwargs: CloudArtifactRepository._parallelized_download_from_cloud(
                cloud_artifact_instance, *args, **kwargs
            )
        )

        # Mock the chunk object
        class FakeChunk:
            def __init__(self, index, start, end, path):
                self.index = index
                self.start = start
                self.end = end
                self.path = path

        fake_chunk_1 = FakeChunk(index=1, start=0, end=100, path="fake_path_1")
        mock_failed_downloads = {fake_chunk_1: "fake_chunk_1"}

        # Wrap fake_chunk_1 in a Future
        future = Future()
        if future_result is None:
            future.set_result(fake_chunk_1)
        else:
            future.set_exception(future_result)

        futures = {future: fake_chunk_1}

        # Create a new ArtifactCredentialInfo object
        fake_credential = ArtifactCredentialInfo(
            signed_uri="fake_signed_uri",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
        )
        fake_credential.headers.extend([
            ArtifactCredentialInfo.HttpHeader(name="fake_header_name", value="fake_header_value")
        ])

        # Set the return value of _get_read_credential_infos to the fake_credential object
        cloud_artifact_instance._get_read_credential_infos.return_value = [fake_credential]

        # Set return value for mocks
        cloud_artifact_instance._get_read_credential_infos.return_value = [fake_credential]
        cloud_artifact_instance._get_uri_for_path.return_value = "fake_uri_path"

        cloud_artifact_instance.chunk_thread_pool.submit.return_value = future

        # Create a fake local path using tmp_path
        fake_local_path = tmp_path / "downloaded_file"

        with (
            mock.patch(
                "mlflow.store.artifact.cloud_artifact_repo.parallelized_download_file_using_http_uri",
                return_value=mock_failed_downloads,
            ),
            mock.patch(
                "mlflow.store.artifact.cloud_artifact_repo.as_completed",
                return_value=futures,
            ),
        ):
            if future_result:
                with pytest.raises(
                    MlflowException, match="All retries have been exhausted. Download has failed."
                ):
                    cloud_artifact_instance._parallelized_download_from_cloud(
                        1, "fake_remote_path", str(fake_local_path)
                    )
            else:
                cloud_artifact_instance._parallelized_download_from_cloud(
                    1, "fake_remote_path", str(fake_local_path)
                )

            assert (
                cloud_artifact_instance._get_read_credential_infos.call_count == expected_call_count
            )
            assert (
                cloud_artifact_instance._get_read_credential_infos.call_count == expected_call_count
            )

            for call in cloud_artifact_instance.chunk_thread_pool.submit.call_args_list:
                assert call == mock.call(
                    mock.ANY,
                    range_start=fake_chunk_1.start,
                    range_end=fake_chunk_1.end,
                    headers=mock.ANY,
                    download_path=str(fake_local_path),
                    http_uri="fake_signed_uri",
                )


# ==================== Upload Pipeline Tests ====================

CLOUD_REPO = "mlflow.store.artifact.cloud_artifact_repo.CloudArtifactRepository"


def _unmock(instance, method_name):
    getattr(instance, method_name).side_effect = lambda *args, **kwargs: getattr(
        CloudArtifactRepository, method_name
    )(instance, *args, **kwargs)


# -- _collect_upload_plans --


def test_collect_upload_plans_flat_directory(tmp_path):
    for name in ["a.txt", "b.txt", "c.txt"]:
        (tmp_path / name).write_text(name)

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_collect_upload_plans")

        plans = instance._collect_upload_plans(str(tmp_path), "artifacts")

    assert len(plans) == 3
    dest_paths = sorted(p.dest_path for p in plans)
    assert dest_paths == ["artifacts/a.txt", "artifacts/b.txt", "artifacts/c.txt"]
    for plan in plans:
        assert plan.credential_info is None
        assert plan.file_size > 0


def test_collect_upload_plans_nested_directories(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "root.txt").write_text("root")
    (sub / "nested.txt").write_text("nested")

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_collect_upload_plans")

        plans = instance._collect_upload_plans(str(tmp_path), "art")

    dest_paths = sorted(p.dest_path for p in plans)
    assert dest_paths == ["art/root.txt", "art/subdir/nested.txt"]


def test_collect_upload_plans_empty_directory(tmp_path):
    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_collect_upload_plans")

        plans = instance._collect_upload_plans(str(tmp_path), "art")

    assert plans == []


# -- _detect_cloud_type --


def test_detect_cloud_type_prefers_small_file():
    large_plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("large.bin", "dest/large.bin"),
        file_size=600 * 1024**2,
    )
    small_plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("small.txt", "dest/small.txt"),
        file_size=1024,
    )
    cred = ArtifactCredentialInfo(
        signed_uri="https://mock", type=ArtifactCredentialType.AWS_PRESIGNED_URL
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_detect_cloud_type")
        instance._get_write_credential_infos.return_value = [cred]

        result = instance._detect_cloud_type([large_plan, small_plan])

    assert result == ArtifactCredentialType.AWS_PRESIGNED_URL
    instance._get_write_credential_infos.assert_called_once_with(["dest/small.txt"])
    assert small_plan.credential_info == cred
    assert large_plan.credential_info is None


def test_detect_cloud_type_falls_back_to_first_when_all_large():
    plan1 = FileUploadPlan(
        staged_upload=StagedArtifactUpload("big1.bin", "dest/big1.bin"),
        file_size=600 * 1024**2,
    )
    plan2 = FileUploadPlan(
        staged_upload=StagedArtifactUpload("big2.bin", "dest/big2.bin"),
        file_size=700 * 1024**2,
    )
    cred = ArtifactCredentialInfo(
        signed_uri="https://mock", type=ArtifactCredentialType.AZURE_SAS_URI
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_detect_cloud_type")
        instance._get_write_credential_infos.return_value = [cred]

        result = instance._detect_cloud_type([plan1, plan2])

    assert result == ArtifactCredentialType.AZURE_SAS_URI
    instance._get_write_credential_infos.assert_called_once_with(["dest/big1.bin"])
    assert plan1.credential_info == cred


def test_detect_cloud_type_empty_plans():
    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_detect_cloud_type")

        result = instance._detect_cloud_type([])

    assert result is None
    instance._get_write_credential_infos.assert_not_called()


def test_detect_cloud_type_no_creds_returned():
    plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("f.txt", "dest/f.txt"),
        file_size=1024,
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_detect_cloud_type")
        instance._get_write_credential_infos.return_value = []

        result = instance._detect_cloud_type([plan])

    assert result is None
    assert plan.credential_info is None


# -- _fetch_credentials_for_plans --


def test_fetch_credentials_skips_cached():
    cached_cred = ArtifactCredentialInfo(signed_uri="https://cached")
    plan_cached = FileUploadPlan(
        staged_upload=StagedArtifactUpload("cached.txt", "dest/cached.txt"),
        file_size=100,
        credential_info=cached_cred,
    )
    plan_needs_cred = FileUploadPlan(
        staged_upload=StagedArtifactUpload("new.txt", "dest/new.txt"),
        file_size=200,
    )
    new_cred = ArtifactCredentialInfo(signed_uri="https://new")

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_fetch_credentials_for_plans")
        instance._get_write_credential_infos.return_value = [new_cred]

        instance._fetch_credentials_for_plans([plan_cached, plan_needs_cred])

    instance._get_write_credential_infos.assert_called_once_with(["dest/new.txt"])
    assert plan_cached.credential_info == cached_cred
    assert plan_needs_cred.credential_info == new_cred


def test_fetch_credentials_all_cached():
    cred = ArtifactCredentialInfo(signed_uri="https://cached")
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=100,
            credential_info=cred,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=200,
            credential_info=cred,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_fetch_credentials_for_plans")

        instance._fetch_credentials_for_plans(plans)

    instance._get_write_credential_infos.assert_not_called()


def test_fetch_credentials_failure_warns():
    plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("f.txt", "dest/f.txt"),
        file_size=100,
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_fetch_credentials_for_plans")
        instance._get_write_credential_infos.side_effect = RuntimeError("network error")

        instance._fetch_credentials_for_plans([plan])

    assert plan.credential_info is None


# -- _upload_files_parallel --


def test_upload_files_parallel_success():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("f1.txt", "dest/f1.txt"),
            file_size=100,
            credential_info=ArtifactCredentialInfo(signed_uri="https://1"),
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("f2.txt", "dest/f2.txt"),
            file_size=200,
            credential_info=ArtifactCredentialInfo(signed_uri="https://2"),
        ),
    ]
    future1 = Future()
    future1.set_result(None)
    future2 = Future()
    future2.set_result(None)

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_files_parallel")
        instance.thread_pool.submit.side_effect = [future1, future2]

        failures = instance._upload_files_parallel(plans)

    assert failures == {}
    assert instance.thread_pool.submit.call_count == 2


def test_upload_files_parallel_partial_failure():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("ok.txt", "dest/ok.txt"),
            file_size=100,
            credential_info=ArtifactCredentialInfo(signed_uri="https://1"),
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("fail.txt", "dest/fail.txt"),
            file_size=200,
            credential_info=ArtifactCredentialInfo(signed_uri="https://2"),
        ),
    ]
    ok_future = Future()
    ok_future.set_result(None)
    fail_future = Future()
    fail_future.set_exception(RuntimeError("upload failed"))

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_files_parallel")
        instance.thread_pool.submit.side_effect = [ok_future, fail_future]

        failures = instance._upload_files_parallel(plans)

    assert "fail.txt" in failures
    assert "ok.txt" not in failures


# -- _upload_files_serially_with_delays --


def test_serial_upload_with_delays_success(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=1024,
            credential_info=ArtifactCredentialInfo(signed_uri="https://1"),
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=1024,
            credential_info=ArtifactCredentialInfo(signed_uri="https://2"),
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("c.txt", "dest/c.txt"),
            file_size=1024,
            credential_info=ArtifactCredentialInfo(signed_uri="https://3"),
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_files_serially_with_delays")

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_files_serially_with_delays(plans)

    assert failures == {}
    assert instance._upload_to_cloud.call_count == 3
    # Sleep called between uploads (after file 0 and file 1), not after last file
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(3.0)


def test_serial_upload_with_delays_failure(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("ok.txt", "dest/ok.txt"),
            file_size=1024,
            credential_info=ArtifactCredentialInfo(signed_uri="https://1"),
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("fail.txt", "dest/fail.txt"),
            file_size=1024,
            credential_info=ArtifactCredentialInfo(signed_uri="https://2"),
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_files_serially_with_delays")
        instance._upload_to_cloud.side_effect = [None, RuntimeError("upload error")]

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_files_serially_with_delays(plans)

    assert "fail.txt" in failures
    assert "ok.txt" not in failures
    # Sleep called after first (successful) file, before second file
    mock_sleep.assert_called_once_with(3.0)


def test_serial_upload_with_delays_single_file(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("only.txt", "dest/only.txt"),
        file_size=1024,
        credential_info=ArtifactCredentialInfo(signed_uri="https://1"),
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_files_serially_with_delays")

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_files_serially_with_delays([plan])

    assert failures == {}
    mock_sleep.assert_not_called()


# -- _upload_batch_aws --


def test_upload_batch_aws_mixed_small_and_large():
    small = FileUploadPlan(
        staged_upload=StagedArtifactUpload("small.txt", "dest/small.txt"),
        file_size=1024,
    )
    large = FileUploadPlan(
        staged_upload=StagedArtifactUpload("large.bin", "dest/large.bin"),
        file_size=600 * 1024**2,
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_aws")
        instance._upload_files_parallel.return_value = {}

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_batch_aws([small, large])

    assert failures == {}
    instance._fetch_credentials_for_plans.assert_called_once_with([small])
    assert instance._upload_files_parallel.call_count == 2
    instance._upload_files_parallel.assert_any_call([small])
    instance._upload_files_parallel.assert_any_call([large])
    mock_sleep.assert_called_once_with(3.0)


def test_upload_batch_aws_only_small_files():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=1024,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=2048,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_aws")
        instance._upload_files_parallel.return_value = {}

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_batch_aws(plans)

    assert failures == {}
    instance._upload_files_parallel.assert_called_once_with(plans)
    mock_sleep.assert_not_called()


def test_upload_batch_aws_only_large_files():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("big.bin", "dest/big.bin"),
            file_size=600 * 1024**2,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_aws")
        instance._upload_files_parallel.return_value = {}

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep") as mock_sleep:
            failures = instance._upload_batch_aws(plans)

    assert failures == {}
    instance._fetch_credentials_for_plans.assert_called_once_with([])
    instance._upload_files_parallel.assert_called_once_with(plans)
    mock_sleep.assert_not_called()


def test_upload_batch_aws_aggregates_failures():
    small = FileUploadPlan(
        staged_upload=StagedArtifactUpload("small.txt", "dest/small.txt"),
        file_size=1024,
    )
    large = FileUploadPlan(
        staged_upload=StagedArtifactUpload("large.bin", "dest/large.bin"),
        file_size=600 * 1024**2,
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_aws")
        instance._upload_files_parallel.side_effect = [
            {"small.txt": "small error"},
            {"large.bin": "large error"},
        ]

        with mock.patch("mlflow.store.artifact.cloud_artifact_repo.time.sleep"):
            failures = instance._upload_batch_aws([small, large])

    assert failures == {"small.txt": "small error", "large.bin": "large error"}


# -- _upload_batch_azure_gcp --


@pytest.mark.parametrize(
    "cloud_type",
    [ArtifactCredentialType.AZURE_SAS_URI, ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI],
)
def test_upload_batch_azure_proxy_safe_serialized(cloud_type):
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=1024,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=2048,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_azure_gcp")
        instance._requires_proxy_safe_uploads = True
        instance._upload_files_serially_with_delays.return_value = {}

        failures = instance._upload_batch_azure_gcp(plans, cloud_type)

    assert failures == {}
    instance._fetch_credentials_for_plans.assert_called_once_with(plans)
    instance._upload_files_serially_with_delays.assert_called_once_with(plans)
    instance._upload_files_parallel.assert_not_called()


def test_upload_batch_azure_proxy_safe_single_file():
    plan = FileUploadPlan(
        staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
        file_size=1024,
    )

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_azure_gcp")
        instance._requires_proxy_safe_uploads = True
        instance._upload_files_parallel.return_value = {}

        failures = instance._upload_batch_azure_gcp([plan], ArtifactCredentialType.AZURE_SAS_URI)

    assert failures == {}
    instance._upload_files_parallel.assert_called_once_with([plan])
    instance._upload_files_serially_with_delays.assert_not_called()


def test_upload_batch_azure_direct_parallel():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=1024,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=2048,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_azure_gcp")
        instance._requires_proxy_safe_uploads = False
        instance._upload_files_parallel.return_value = {}

        failures = instance._upload_batch_azure_gcp(plans, ArtifactCredentialType.AZURE_SAS_URI)

    assert failures == {}
    instance._upload_files_parallel.assert_called_once_with(plans)
    instance._upload_files_serially_with_delays.assert_not_called()


def test_upload_batch_gcp_always_parallel():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=1024,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=2048,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_azure_gcp")
        instance._requires_proxy_safe_uploads = True
        instance._upload_files_parallel.return_value = {}

        failures = instance._upload_batch_azure_gcp(plans, ArtifactCredentialType.GCP_SIGNED_URL)

    assert failures == {}
    instance._upload_files_parallel.assert_called_once_with(plans)
    instance._upload_files_serially_with_delays.assert_not_called()


# -- _upload_batch_generic --


def test_upload_batch_generic():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("f.txt", "dest/f.txt"),
            file_size=100,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "_upload_batch_generic")
        instance._upload_files_parallel.return_value = {}

        failures = instance._upload_batch_generic(plans)

    assert failures == {}
    instance._fetch_credentials_for_plans.assert_called_once_with(plans)
    instance._upload_files_parallel.assert_called_once_with(plans)


# -- log_artifacts integration --


def test_log_artifacts_empty_directory():
    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "log_artifacts")
        instance._collect_upload_plans.return_value = []

        instance.log_artifacts("/some/dir")

    instance._detect_cloud_type.assert_not_called()
    instance._upload_batch_aws.assert_not_called()
    instance._upload_batch_azure_gcp.assert_not_called()
    instance._upload_batch_generic.assert_not_called()


@pytest.mark.parametrize(
    ("cloud_type", "expected_method"),
    [
        (ArtifactCredentialType.AWS_PRESIGNED_URL, "_upload_batch_aws"),
        (ArtifactCredentialType.AZURE_SAS_URI, "_upload_batch_azure_gcp"),
        (ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI, "_upload_batch_azure_gcp"),
        (ArtifactCredentialType.GCP_SIGNED_URL, "_upload_batch_generic"),
        (None, "_upload_batch_generic"),
    ],
)
def test_log_artifacts_routes_by_cloud_type(cloud_type, expected_method):
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("f.txt", "dest/f.txt"),
            file_size=100,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "log_artifacts")
        instance._collect_upload_plans.return_value = plans
        instance._detect_cloud_type.return_value = cloud_type
        for method in ("_upload_batch_aws", "_upload_batch_azure_gcp", "_upload_batch_generic"):
            getattr(instance, method).return_value = {}

        instance.log_artifacts("/some/dir", "art")

    getattr(instance, expected_method).assert_called_once()


def test_log_artifacts_batches_uploads():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload(f"f{i}.txt", f"dest/f{i}.txt"),
            file_size=100,
        )
        for i in range(_ARTIFACT_UPLOAD_BATCH_SIZE * 2 + 20)
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "log_artifacts")
        instance._collect_upload_plans.return_value = plans
        instance._detect_cloud_type.return_value = ArtifactCredentialType.AWS_PRESIGNED_URL
        instance._upload_batch_aws.return_value = {}

        instance.log_artifacts("/some/dir")

    assert instance._upload_batch_aws.call_count == 3
    batch_sizes = [len(call.args[0]) for call in instance._upload_batch_aws.call_args_list]
    assert batch_sizes == [_ARTIFACT_UPLOAD_BATCH_SIZE, _ARTIFACT_UPLOAD_BATCH_SIZE, 20]


def test_log_artifacts_accumulates_failures():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("f.txt", "dest/f.txt"),
            file_size=100,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "log_artifacts")
        instance.artifact_uri = "dbfs:/artifacts"
        instance._collect_upload_plans.return_value = plans
        instance._detect_cloud_type.return_value = None
        instance._upload_batch_generic.return_value = {"f.txt": "upload error"}

        with pytest.raises(MlflowException, match="failures occurred while uploading"):
            instance.log_artifacts("/some/dir")


def test_log_artifacts_handles_batch_exception():
    plans = [
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("a.txt", "dest/a.txt"),
            file_size=100,
        ),
        FileUploadPlan(
            staged_upload=StagedArtifactUpload("b.txt", "dest/b.txt"),
            file_size=200,
        ),
    ]

    with mock.patch(CLOUD_REPO) as mock_cls:
        instance = mock_cls.return_value
        _unmock(instance, "log_artifacts")
        instance.artifact_uri = "dbfs:/artifacts"
        instance._collect_upload_plans.return_value = plans
        instance._detect_cloud_type.return_value = None
        instance._upload_batch_generic.side_effect = RuntimeError("batch crash")

        with pytest.raises(MlflowException, match="failures occurred while uploading"):
            instance.log_artifacts("/some/dir")


# -- _requires_proxy_safe_uploads --


def test_requires_proxy_safe_uploads_default_false():
    assert CloudArtifactRepository._requires_proxy_safe_uploads.fget(mock.MagicMock()) is False

import io
import pickle
import random
import threading
import time

import pytest
from PIL import Image

from mlflow import MlflowException
from mlflow.utils.async_logging.async_artifacts_logging_queue import AsyncArtifactsLoggingQueue

TOTAL_ARTIFACTS = 5


class RunArtifacts:
    def __init__(self, throw_exception_on_artifact_number=None):
        if throw_exception_on_artifact_number is None:
            throw_exception_on_artifact_number = []
        self.received_run_id = ""
        self.received_artifacts = []
        self.received_filenames = []
        self.received_artifact_paths = []
        self.artifact_count = 0
        self.throw_exception_on_artifact_number = (
            throw_exception_on_artifact_number if throw_exception_on_artifact_number else []
        )

    def consume_queue_data(self, filename, artifact_path, artifact):
        self.artifact_count += 1
        if self.artifact_count in self.throw_exception_on_artifact_number:
            raise MlflowException("Failed to log run data")
        self.received_artifacts.append(artifact)
        self.received_filenames.append(filename)
        self.received_artifact_paths.append(artifact_path)


def _get_run_artifacts(total_artifacts=TOTAL_ARTIFACTS):
    for num in range(0, total_artifacts):
        filename = f"image_{num}.png"
        artifact_path = f"images/artifact_{num}"
        artifact = Image.new("RGB", (100, 100), color="red")
        yield filename, artifact_path, artifact


def _assert_sent_received_artifacts(
    filenames_sent,
    artifact_paths_sent,
    artifacts_sent,
    received_filenames,
    received_artifact_paths,
    received_artifacts,
):
    for num in range(1, len(filenames_sent)):
        assert filenames_sent[num] == received_filenames[num]

    for num in range(1, len(artifact_paths_sent)):
        assert artifact_paths_sent[num] == received_artifact_paths[num]

    for num in range(1, len(artifacts_sent)):
        assert artifacts_sent[num] == received_artifacts[num]


def test_single_thread_publish_consume_queue():
    run_artifacts = RunArtifacts()
    async_logging_queue = AsyncArtifactsLoggingQueue(run_artifacts.consume_queue_data)
    async_logging_queue.activate()
    filenames_sent = []
    artifact_paths_sent = []
    artifacts_sent = []
    for filename, artifact_path, artifact in _get_run_artifacts():
        async_logging_queue.log_artifacts_async(
            filename=filename, artifact_path=artifact_path, artifact=artifact
        )
        filenames_sent.append(filename)
        artifact_paths_sent.append(artifact_path)
        artifacts_sent.append(artifact)
    async_logging_queue.flush()

    _assert_sent_received_artifacts(
        filenames_sent,
        artifact_paths_sent,
        artifacts_sent,
        run_artifacts.received_filenames,
        run_artifacts.received_artifact_paths,
        run_artifacts.received_artifacts,
    )


def test_queue_activation():
    run_artifacts = RunArtifacts()
    async_logging_queue = AsyncArtifactsLoggingQueue(run_artifacts.consume_queue_data)

    assert not async_logging_queue._is_activated

    for filename, artifact_path, artifact in _get_run_artifacts(1):
        with pytest.raises(MlflowException, match="AsyncArtifactsLoggingQueue is not activated."):
            async_logging_queue.log_artifacts_async(
                filename=filename, artifact_path=artifact_path, artifact=artifact
            )

    async_logging_queue.activate()
    assert async_logging_queue._is_activated


def test_partial_logging_failed():
    run_data = RunArtifacts(throw_exception_on_artifact_number=[3, 4])

    async_logging_queue = AsyncArtifactsLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()

    filenames_sent = []
    artifact_paths_sent = []
    artifacts_sent = []

    run_operations = []
    batch_id = 1
    for filename, artifact_path, artifact in _get_run_artifacts():
        if batch_id in [3, 4]:
            with pytest.raises(MlflowException, match="Failed to log run data"):
                async_logging_queue.log_artifacts_async(
                    filename=filename, artifact_path=artifact_path, artifact=artifact
                ).wait()
        else:
            run_operations.append(
                async_logging_queue.log_artifacts_async(
                    filename=filename, artifact_path=artifact_path, artifact=artifact
                )
            )
            filenames_sent.append(filename)
            artifact_paths_sent.append(artifact_path)
            artifacts_sent.append(artifact)

        batch_id += 1

    for run_operation in run_operations:
        run_operation.wait()

    _assert_sent_received_artifacts(
        filenames_sent,
        artifact_paths_sent,
        artifacts_sent,
        run_data.received_filenames,
        run_data.received_artifact_paths,
        run_data.received_artifacts,
    )


def test_publish_multithread_consume_single_thread():
    run_data = RunArtifacts(throw_exception_on_artifact_number=[])

    async_logging_queue = AsyncArtifactsLoggingQueue(run_data.consume_queue_data)
    async_logging_queue.activate()

    def _send_artifact(run_data_queueing_processor, run_operations=None):
        if run_operations is None:
            run_operations = []
        filenames_sent = []
        artifact_paths_sent = []
        artifacts_sent = []

        for filename, artifact_path, artifact in _get_run_artifacts():
            run_operations.append(
                run_data_queueing_processor.log_artifacts_async(
                    filename=filename, artifact_path=artifact_path, artifact=artifact
                )
            )

            time.sleep(random.randint(1, 3))
            filenames_sent.append(filename)
            artifact_paths_sent.append(artifact_path)
            artifacts_sent.append(artifact)

    run_operations = []
    t1 = threading.Thread(target=_send_artifact, args=(async_logging_queue, run_operations))
    t2 = threading.Thread(target=_send_artifact, args=(async_logging_queue, run_operations))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for run_operation in run_operations:
        run_operation.wait()

    assert len(run_data.received_filenames) == 2 * TOTAL_ARTIFACTS
    assert len(run_data.received_artifact_paths) == 2 * TOTAL_ARTIFACTS
    assert len(run_data.received_artifacts) == 2 * TOTAL_ARTIFACTS


class Consumer:
    def __init__(self) -> None:
        self.filenames = []
        self.artifact_paths = []
        self.artifacts = []

    def consume_queue_data(self, filename, artifact_path, artifact):
        time.sleep(0.5)
        self.filenames.append(filename)
        self.artifact_paths.append(artifact_path)
        self.artifacts.append(artifact)


def test_async_logging_queue_pickle():
    consumer = Consumer()
    async_logging_queue = AsyncArtifactsLoggingQueue(consumer.consume_queue_data)

    # Pickle the queue without activating it.
    buffer = io.BytesIO()
    pickle.dump(async_logging_queue, buffer)
    deserialized_queue = pickle.loads(buffer.getvalue())  # Type: AsyncArtifactsLoggingQueue

    # activate the queue and then try to pickle it
    async_logging_queue.activate()

    run_operations = []
    for val in range(0, 10):
        run_operations.append(
            async_logging_queue.log_artifacts_async(
                filename=f"image-{val}.png",
                artifact_path="images/image-artifact.png",
                artifact=Image.new("RGB", (100, 100), color="blue"),
            )
        )

    # Pickle the queue
    buffer = io.BytesIO()
    pickle.dump(async_logging_queue, buffer)

    deserialized_queue = pickle.loads(buffer.getvalue())  # Type: AsyncLoggingQueue
    assert deserialized_queue._queue.empty()
    assert deserialized_queue._lock is not None
    assert deserialized_queue._is_activated is False

    for run_operation in run_operations:
        run_operation.wait()

    assert len(consumer.filenames) == 10

    # try to log using deserialized queue after activating it.
    deserialized_queue.activate()
    assert deserialized_queue._is_activated

    run_operations = []

    for val in range(0, 10):
        run_operations.append(
            deserialized_queue.log_artifacts_async(
                filename=f"image2-{val}.png",
                artifact_path="images/image-artifact2.png",
                artifact=Image.new("RGB", (100, 100), color="green"),
            )
        )

    for run_operation in run_operations:
        run_operation.wait()

    assert len(deserialized_queue._artifact_logging_func.__self__.filenames) == 10

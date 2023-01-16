import os
import posixpath
import tempfile
from abc import abstractmethod, ABCMeta
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import developer_stable
from mlflow.utils.validation import path_not_unique, bad_path_message


# Constants used to determine max level of parallelism to use while uploading/downloading artifacts.
# Max threads to use for parallelism.
_NUM_MAX_THREADS = 8
# Max threads per CPU
_NUM_MAX_THREADS_PER_CPU = 2
assert _NUM_MAX_THREADS >= _NUM_MAX_THREADS_PER_CPU
assert _NUM_MAX_THREADS_PER_CPU > 0
# Default number of CPUs to assume on the machine if unavailable to fetch it using os.cpu_count()
_NUM_DEFAULT_CPUS = _NUM_MAX_THREADS // _NUM_MAX_THREADS_PER_CPU


@developer_stable
class ArtifactRepository:
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri
        # Limit the number of threads used for artifact uploads/downloads. Use at most
        # constants._NUM_MAX_THREADS threads or 2 * the number of CPU cores available on the
        # system (whichever is smaller)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact.
        """
        pass

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        pass

    @abstractmethod
    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contains desired artifacts

        :return: List of artifacts as FileInfo listed directly under path.
        """
        pass

    def _is_directory(self, artifact_path):
        listing = self.list_artifacts(artifact_path)
        return len(listing) > 0

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        """
        Creates a local filesystem location to be used as a destination for downloading the artifact
        specified by `src_artifact_path`. The destination location is a subdirectory of the
        specified `dst_local_dir_path`, which is determined according to the structure of
        `src_artifact_path`. For example, if `src_artifact_path` is `dir1/file1.txt`, then the
        resulting destination path is `<dst_local_dir_path>/dir1/file1.txt`. Local directories are
        created for the resulting destination location if they do not exist.

        :param src_artifact_path: A relative, POSIX-style path referring to an artifact stored
                                  within the repository's artifact root location.
                                  `src_artifact_path` should be specified relative to the
                                  repository's artifact root location.
        :param dst_local_dir_path: The absolute path to a local filesystem directory in which the
                                   local destination path will be contained. The local destination
                                   path may be contained in a subdirectory of `dst_root_dir` if
                                   `src_artifact_path` contains subdirectories.
        :return: The absolute path to a local filesystem location to be used as a destination
                 for downloading the artifact specified by `src_artifact_path`.
        """
        src_artifact_path = src_artifact_path.rstrip("/")  # Ensure correct dirname for trailing '/'
        dirpath = posixpath.dirname(src_artifact_path)
        local_dir_path = os.path.join(dst_local_dir_path, dirpath)
        local_file_path = os.path.join(dst_local_dir_path, src_artifact_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        return local_file_path

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        :param artifact_path: Relative source path to the desired artifacts.
        :param dst_path: Absolute path of the local filesystem destination directory to which to
                         download the specified artifacts. This directory must already exist.
                         If unspecified, the artifacts will either be downloaded to a new
                         uniquely-named directory on the local filesystem or will be returned
                         directly in the case of the LocalArtifactRepository.

        :return: Absolute path of the local filesystem location containing the desired artifacts.
        """
        # Represents an in-progress file artifact download to a local filesystem location
        InflightDownload = namedtuple(
            "InflightDownload",
            [
                # The artifact path, given relative to the repository's artifact root location
                "src_artifact_path",
                # The local filesystem destination path to which artifacts are being downloaded
                "dst_local_path",
                # A future representing the artifact download operation
                "download_future",
            ],
        )

        def async_download_artifact(src_artifact_path, dst_local_dir_path):
            """
            Download the file artifact specified by `src_artifact_path` to the local filesystem
            directory specified by `dst_local_dir_path`.
            :param src_artifact_path: A relative, POSIX-style path referring to a file artifact
                                      stored within the repository's artifact root location.
                                      `src_artifact_path` should be specified relative to the
                                      repository's artifact root location.
            :param dst_local_dir_path: Absolute path of the local filesystem destination directory
                                       to which to download the specified artifact. The downloaded
                                       artifact may be written to a subdirectory of
                                       `dst_local_dir_path` if `src_artifact_path` contains
                                       subdirectories.
            :return: A local filesystem path referring to the downloaded file.
            """
            inflight_downloads = []
            local_destination_file_path = self._create_download_destination(
                src_artifact_path=src_artifact_path, dst_local_dir_path=dst_local_dir_path
            )
            download_future = self.thread_pool.submit(
                self._download_file,
                remote_file_path=src_artifact_path,
                local_path=local_destination_file_path,
            )
            inflight_downloads.append(
                InflightDownload(
                    src_artifact_path=src_artifact_path,
                    dst_local_path=local_destination_file_path,
                    download_future=download_future,
                )
            )
            return inflight_downloads

        def async_download_artifact_dir(src_artifact_dir_path, dst_local_dir_path):
            """
            Initiate an asynchronous download of the artifact directory specified by
            `src_artifact_dir_path` to the local filesystem directory specified by
            `dst_local_dir_path`.

            This implementation is adapted from
            https://github.com/mlflow/mlflow/blob/a776b54fa8e1beeca6a984864c6375e9ed38f8c0/mlflow/
            store/artifact/artifact_repo.py#L93.

            :param src_artifact_dir_path: A relative, POSIX-style path referring to a directory of
                                          of artifacts stored within the repository's artifact root
                                          location. `src_artifact_dir_path` should be specified
                                          relative to the repository's artifact root location.
            :param dst_local_dir_path: Absolute path of the local filesystem destination directory
                                       to which to download the specified artifact directory. The
                                       downloaded artifacts may be written to a subdirectory of
                                       `dst_local_dir_path` if `src_artifact_dir_path` contains
                                       subdirectories.
            :return: A tuple whose first element is the destination directory of the downloaded
                     artifacts on the local filesystem and whose second element is a list of
                     `InflightDownload` objects, each of which represents an inflight asynchronous
                     download operation for a file in the specified artifact directory.
            """
            local_dir = os.path.join(dst_local_dir_path, src_artifact_dir_path)
            inflight_downloads = []
            dir_content = [  # prevent infinite loop, sometimes the dir is recursively included
                file_info
                for file_info in self.list_artifacts(src_artifact_dir_path)
                if file_info.path != "." and file_info.path != src_artifact_dir_path
            ]
            if not dir_content:  # empty dir
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
            else:
                for file_info in dir_content:
                    if file_info.is_dir:
                        inflight_downloads += async_download_artifact_dir(
                            src_artifact_dir_path=file_info.path,
                            dst_local_dir_path=dst_local_dir_path,
                        )[1]
                    else:
                        inflight_downloads += async_download_artifact(
                            src_artifact_path=file_info.path,
                            dst_local_dir_path=dst_local_dir_path,
                        )
            return local_dir, inflight_downloads

        if dst_path is None:
            dst_path = tempfile.mkdtemp()
        dst_path = os.path.abspath(dst_path)

        if not os.path.exists(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts does not"
                    f" exist! Destination path: {dst_path}"
                ),
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        elif not os.path.isdir(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts must be a directory!"
                    f" Destination path: {dst_path}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Check if the artifacts points to a directory
        if self._is_directory(artifact_path):
            dst_local_path, inflight_downloads = async_download_artifact_dir(
                src_artifact_dir_path=artifact_path, dst_local_dir_path=dst_path
            )
        else:
            inflight_downloads = async_download_artifact(
                src_artifact_path=artifact_path, dst_local_dir_path=dst_path
            )
            assert (
                len(inflight_downloads) == 1
            ), "Expected one inflight download for a file artifact, got {} downloads".format(
                len(inflight_downloads)
            )
            dst_local_path = inflight_downloads[0].dst_local_path

        # Join futures to ensure that all artifacts have been downloaded prior to returning
        failed_downloads = {}
        for inflight_download in inflight_downloads:
            try:
                inflight_download.download_future.result()
            except Exception as e:
                failed_downloads[inflight_download.src_artifact_path] = repr(e)

        if len(failed_downloads) > 0:
            raise MlflowException(
                message=(
                    "The following failures occurred while downloading one or more"
                    " artifacts from {artifact_root}: {failures}".format(
                        artifact_root=self.artifact_uri,
                        failures=failed_downloads,
                    )
                )
            )
        return dst_local_path

    @abstractmethod
    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        :param remote_file_path: Source path to the remote file, relative to the root
                                 directory of the artifact repository.
        :param local_path: The path to which to save the downloaded file.
        """
        pass

    def delete_artifacts(self, artifact_path=None):
        """
        Delete the artifacts at the specified location.
        Supports the deletion of a single file or of a directory. Deletion of a directory
        is recursive.
        :param artifact_path: Path of the artifact to delete
        """
        pass

    @property
    def max_workers(self) -> int:
        """Compute the number of workers to use for multi-threading."""
        num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
        return min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)


def verify_artifact_path(artifact_path):
    if artifact_path and path_not_unique(artifact_path):
        raise MlflowException(
            "Invalid artifact path: '{}'. {}".format(artifact_path, bad_path_message(artifact_path))
        )

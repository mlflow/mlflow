import os
import posixpath
import tempfile
from abc import abstractmethod, ABCMeta

from mlflow.utils.validation import path_not_unique, bad_path_message

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST


class ArtifactRepository:
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri

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

        # TODO: Probably need to add a more efficient method to stream just a single artifact
        #       without downloading it, or to get a pre-signed URL for cloud storage.

        def download_artifacts_into(artifact_path, dest_dir):
            basename = posixpath.basename(artifact_path)
            local_path = os.path.join(dest_dir, basename)
            listing = self.list_artifacts(artifact_path)
            if len(listing) > 0:
                # Artifact_path is a directory, so make a directory for it and download everything
                if not os.path.exists(local_path):
                    os.mkdir(local_path)
                for file_info in listing:
                    # prevent an infinite loop (sometimes the current path is listed e.g. as ".")
                    if file_info.path == "." or file_info.path == artifact_path:
                        continue
                    download_artifacts_into(artifact_path=file_info.path, dest_dir=local_path)
            else:
                self._download_file(remote_file_path=artifact_path, local_path=local_path)
            return local_path

        if dst_path is None:
            dst_path = tempfile.mkdtemp()
        dst_path = os.path.abspath(dst_path)

        if not os.path.exists(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts does not"
                    " exist! Destination path: {dst_path}".format(dst_path=dst_path)),
                error_code=RESOURCE_DOES_NOT_EXIST)
        elif not os.path.isdir(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts must be a directory!"
                    " Destination path: {dst_path}".format(dst_path=dst_path)),
                error_code=INVALID_PARAMETER_VALUE)

        return download_artifacts_into(artifact_path, dst_path)

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


def verify_artifact_path(artifact_path):
    if artifact_path and path_not_unique(artifact_path):
        raise MlflowException("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                   bad_path_message(artifact_path)))

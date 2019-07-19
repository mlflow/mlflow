import logging
import os
import posixpath
import tempfile
from abc import abstractmethod, ABCMeta
import sqlalchemy
from contextlib import contextmanager

from alembic.script import ScriptDirectory
from mlflow.utils.validation import path_not_unique, bad_path_message
from mlflow.utils import extract_db_type_from_uri
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.dbmodels.initial_artifact_store_models import Base as InitialBase
from mlflow.store.dbmodels.initial_artifact_store_models import Base, SqlArtifact
from mlflow.store.db.utils import _upgrade_db, _get_alembic_config, _get_schema_version
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR
from sqlalchemy import select
_logger = logging.getLogger(__name__)


def _relative_path(base_dir, subdir_path, path_module):
    relative_path = path_module.relpath(subdir_path, base_dir)
    return relative_path if relative_path is not '.' else None


def _relative_path_local(base_dir, subdir_path):
    rel_path = _relative_path(base_dir, subdir_path, os.path)
    return relative_path_to_artifact_path(rel_path) if rel_path is not None else None


class DBArtifactRepository(ArtifactRepository):
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, db_uri):
        self.artifact_uri = db_uri
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = sqlalchemy.create_engine(db_uri)
        super(DBArtifactRepository, self).__init__(db_uri)

        insp = sqlalchemy.inspect(self.engine)
        expected_tables = set([
            SqlArtifact.__tablename__,
        ])
        if len(expected_tables & set(insp.get_table_names())) == 0:
            DBArtifactRepository._initialize_tables(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = self._get_managed_session_maker(SessionMaker)

    @staticmethod
    def _initialize_tables(engine):
        _logger.info("Creating initial MLflow database tables...")
        InitialBase.metadata.create_all(engine)


    @staticmethod
    def _get_managed_session_maker(SessionMaker):
        """
        Creates a factory for producing exception-safe SQLAlchemy sessions that are made available
        using a context manager. Any session produced by this factory is automatically committed
        if no exceptions are encountered within its associated context. If an exception is
        encountered, the session is rolled back. Finally, any session produced by this factory is
        automatically closed when the session's associated context is exited.
        """

        @contextmanager
        def make_managed_session():
            """Provide a transactional scope around a series of operations."""
            session = SessionMaker()
            try:
                yield session
                session.commit()
            except MlflowException:
                session.rollback()
                raise
            except Exception as e:
                session.rollback()
                raise MlflowException(message=e, error_code=INTERNAL_ERROR)
            finally:
                session.close()

        return make_managed_session

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

        _, file_name = os.path.split(local_file)
        with self.ManagedSessionMaker() as session:
            artifact = SqlArtifact(
                artifact_name=file_name, group_name=artifact_path, artifact_content=open(
                    local_file, "rb").read(),
            )
            session.add(artifact)
            session.flush()
            return str(artifact.artifact_id)

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """

        with self.ManagedSessionMaker() as session:
            for subdir_path, _, files in os.walk(local_dir):

                relative_path = _relative_path_local(local_dir, subdir_path)

                db_subdir_path = posixpath.join(artifact_path, relative_path) \
                    if relative_path else artifact_path

                for each_file in files:
                    source = os.path.join(subdir_path, each_file)
                    destination = posixpath.join(db_subdir_path, each_file)
                    artifact = SqlArtifact(
                        artifact_name=each_file, group_name=db_subdir_path,
                        artifact_content=open(source, "rb").read(),
                    )
                    session.add(artifact)
        session.flush()

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
        # without downloading it, or to get a pre-signed URL for cloud storage.

        def download_artifacts_into(artifact_path, dest_dir):
            basename = posixpath.basename(artifact_path)
            local_path = os.path.join(dest_dir, basename)
            listing = self.list_artifacts(artifact_path)
            if len(listing) > 0:
                # Artifact_path is a directory, so make a directory for it and download everything
                if not os.path.exists(local_path):
                    os.mkdir(local_path)
                for file_info in listing:
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
                                                                       bad_path_message(
                                                                           artifact_path)))



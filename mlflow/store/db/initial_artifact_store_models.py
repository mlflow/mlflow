# Snapshot of MLflow DB models as of the 0.9.1 release, prior to the first database migration.
# Used to standardize initial database state.
# Copied with modifications from
# https://github.com/mlflow/mlflow/blob/v0.9.1/mlflow/store/dbmodels/models.py, which
# is the first database schema that users could be running. In particular, modifications have
# been made to substitute constants from MLflow with hard-coded values (e.g. replacing
# SourceType.to_string(SourceType.NOTEBOOK) with the constant "NOTEBOOK") and ensure
# that all constraint names are unique. Note that pre-1.0 database schemas did not have unique
# constraint names - we provided a one-time migration script for pre-1.0 users so that their
# database schema matched the schema in this file.

from sqlalchemy import (
    Column, String, VARBINARY, BigInteger, Integer, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base

from mlflow.entities import FileInfo
import os

Base = declarative_base()


class SqlArtifact(Base):
    """
    DB model for :py:class:`mlflow.entities.Artifact`. These are recorded in ``artifact`` table.
    """
    __tablename__ = 'artifacts'
    artifact_id = Column(Integer, autoincrement=True)
    """
    Artifact ID: `Integer`. *Primary Key* for ``artifact`` table.
    """
    artifact_name = Column(String(256), nullable=False)
    """
    Artifact Name: ``String` (limit 256 characters).
    """
    group_path = Column(String(256), nullable=False)
    """
    Group path: `String` (limit 256 characters).
    """
    artifact_content = Column(VARBINARY, nullable=False)
    """
    Artifact : `VarBinary`. Defined as *Non null* in table schema.
    """
    artifact_initial_size = Column(BigInteger, nullable=True)
    """
    Artifact Initial Size : `BigInteger`. Defined as *null* in table schema.
    """
    __table_args__ = (
        PrimaryKeyConstraint('artifact_id', name='artifact_pk'),
    )

    def to_file_info(self, base_path):
        """
        Convert DB model to corresponding FileInfo object.

        :return: :py:class:`mlflow.entities.FileInfo`.
        """
        return FileInfo(
            path=os.path.relpath(path=os.path.join(self.group_path, self.artifact_name),
                                 start=base_path),
            is_dir=False,
            file_size=self.artifact_initial_size)

    def __repr__(self):
        return '<SqlArtifact ({}, {}, {}, {}, {})>'.format(self.artifact_id, self.artifact_name,
                                                           self.group_path,
                                                           self.artifact_content,
                                                           self.artifact_initial_size)

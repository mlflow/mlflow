import os
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class ExperimentPermission(Base):
    __tablename__ = "experiment_permissions"
    __table_args__ = (sqlalchemy.UniqueConstraint("experiment_id", "user"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(length=255))
    user = Column(String(length=255))
    permission = Column(String(length=255))


DB_PATH = "permissions.db"


def create_engine():
    return sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")


def init_db():
    if os.path.exists(DB_PATH):
        os.unlink(DB_PATH)
    Base.metadata.create_all(bind=create_engine())


def list_permissions(experiment_id):
    session = sessionmaker(bind=create_engine())()
    permissions = session.query(ExperimentPermission).filter_by(experiment_id=experiment_id).all()
    return permissions


def get_permission(user, experiment_id):
    session = sessionmaker(bind=create_engine())()
    permission = (
        session.query(ExperimentPermission)
        .filter_by(user=user, experiment_id=experiment_id)
        .first()
    )
    return permission


def create_permission(user, experiment_id, permission):
    session = sessionmaker(bind=create_engine())()
    session.add(ExperimentPermission(user=user, experiment_id=experiment_id, permission=permission))
    session.commit()


def update_permission(user, experiment_id, permission):
    session = sessionmaker(bind=create_engine())()
    perm = (
        session.query(ExperimentPermission)
        .filter_by(user=user, experiment_id=experiment_id)
        .first()
    )
    if permission is None:
        raise Exception("Permission not found")
    perm.permission = permission
    session.commit()


def delete_permission(user, experiment_id):
    session = sessionmaker(bind=create_engine())()
    permission = (
        session.query(ExperimentPermission)
        .filter_by(user=user, experiment_id=experiment_id)
        .first()
    )
    if permission is None:
        raise Exception("Permission not found")
    session.delete(permission)
    session.commit()

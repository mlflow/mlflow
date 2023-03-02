import os
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class Permission(Base):
    __tablename__ = "permissions"
    __table_args__ = (sqlalchemy.UniqueConstraint("resource", "user", "key"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    resource = Column(String(length=255))
    user = Column(String(length=255))
    key = Column(String(length=255))
    access_level = Column(String(length=255))


DB_PATH = "permissions.db"


def create_engine():
    return sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")


def init_db():
    os.unlink(DB_PATH)
    Base.metadata.create_all(bind=create_engine())


def get(user, resource, key):
    session = sessionmaker(bind=create_engine())()
    permission = session.query(Permission).filter_by(resource=resource, user=user, key=key).first()
    return permission


def create(user, resource, key, access_level):
    session = sessionmaker(bind=create_engine())()
    session.add(Permission(resource=resource, user=user, key=key, access_level=access_level))
    session.commit()


def update(user, resource, key, access_level):
    session = sessionmaker(bind=create_engine())()
    permission = session.query(Permission).filter_by(resource=resource, user=user, key=key).first()
    permission.access_level = access_level
    session.commit()


def delete(user, resource, key):
    session = sessionmaker(bind=create_engine())()
    permission = session.query(Permission).filter_by(resource=resource, user=user, key=key).first()
    if permission is None:
        return
    session.delete(permission)
    session.commit()

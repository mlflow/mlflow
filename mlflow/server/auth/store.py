from typing import List, Optional
from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy.pagination import Pagination
from werkzeug.security import generate_password_hash, check_password_hash


db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False)
    experiment_permissions = db.relationship("ExperimentPermission", backref="users")
    registered_models_permissions = db.relationship("RegisteredModelPermission", backref="users")


class ExperimentPermission(db.Model):
    __tablename__ = "experiment_permissions"
    id = db.Column(db.Integer(), primary_key=True)
    experiment_id = db.Column(db.String(255), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    permission = db.Column(db.String(255))

    def to_json(self):
        return {
            "experiment_id": self.experiment_id,
            "user_id": self.user_id,
            "permission": self.permission,
        }


class RegisteredModelPermission(db.Model):
    __tablename__ = "registered_models_permissions"
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    permission = db.Column(db.String(255))


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        db.session.commit()


def authenticate_user(username: str, password: str) -> bool:
    user = get_user(username)
    if user is None:
        return False
    pwhash = user.password
    return check_password_hash(pwhash, password)


def create_user(username: str, password: str, is_admin: bool = False):
    pwhash = generate_password_hash(password)
    user = User(username=username, password=pwhash, is_admin=is_admin)
    db.session.add(user)
    db.session.commit()


def has_user(username: str) -> bool:
    return get_user(username) is not None


def get_user(username: str) -> Optional[User]:
    row = db.session.execute(db.select(User).filter_by(username=username)).one_or_none()
    return row.User if row else None


def list_users() -> List[User]:
    rows = db.session.execute(db.select(User)).all()
    return [row.User for row in rows]


def get_experiment_permission(experiment_id: str, user_id: int) -> ExperimentPermission:
    return db.session.execute(
        db.select(ExperimentPermission).filter_by(experiment_id=experiment_id, user_id=user_id)
    ).first()


def get_readable_experiments(user_id: int) -> List[ExperimentPermission]:
    return db.session.execute(db.select(ExperimentPermission).filter_by(user_id=user_id)).all()

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from typing import List

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    admin = db.Column(db.Boolean(), default=False)
    experiment_permissions = db.relationship("ExperimentPermission", backref="users")
    registered_models_permissions = db.relationship("RegisteredModelPermission", backref="users")


class ExperimentPermission(db.Model):
    __tablename__ = "experiment_permissions"
    id = db.Column(db.Integer(), primary_key=True)
    experiment_id = db.Column(db.Integer())
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    permission = db.Column(db.String(255))


class RegisteredModelPermission(db.Model):
    __tablename__ = "registered_models_permissions"
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.Integer())
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    permission = db.Column(db.String(255))


def init_app(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mlflow-auth.db"
    db.init_app(app)


def init_tables():
    db.create_all()
    if User.query.filter_by(admin=True).first() is None:
        db.session.add(
            User(email="admin", password=generate_password_hash("pass_admin"), admin=True)
        )
    if User.query.filter_by(email="user_a@test.com").first() is None:
        db.session.add(User(email="user_a@test.com", password=generate_password_hash("pass_a")))
    if User.query.filter_by(email="user_b@test.com").first() is None:
        db.session.add(User(email="user_b@test.com", password=generate_password_hash("pass_b")))
    db.session.commit()


def create_user(email, password) -> None:
    db.session.add(User(email=email, password=generate_password_hash(password)))
    db.session.commit()


def get_user_by_email(email) -> User:
    return User.query.filter_by(email=email).one_or_404()


def list_experiment_permissions(experiment_id) -> List[ExperimentPermission]:
    return ExperimentPermission.query.filter_by(experiment_id=experiment_id).all()


def get_experiment_permission(experiment_id, user_id) -> ExperimentPermission:
    return ExperimentPermission.query.filter_by(
        experiment_id=experiment_id, user_id=user_id
    ).first()


def create_experiment_permission(experiment_id, user_id, permission) -> None:
    db.session.add(
        ExperimentPermission(experiment_id=experiment_id, user_id=user_id, permission=permission)
    )
    db.session.commit()


def update_experiment_permission(experiment_id, user_id, permission) -> None:
    perm = get_experiment_permission(experiment_id, user_id)
    perm.permission = permission
    db.session.commit()


def delete_experiment_permission(experiment_id, user_id) -> None:
    perm = get_experiment_permission(experiment_id, user_id)
    db.session.delete(perm)
    db.session.commit()

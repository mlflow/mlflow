from typing import List
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash


db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(255), unique=True)
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


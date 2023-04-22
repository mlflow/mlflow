class User:
    def __init__(
        self,
        username,
        password_hash,
        is_admin,
        experiment_permissions=None,
        registered_model_permissions=None,
    ):
        self._username = username
        self._password_hash = password_hash
        self._is_admin = is_admin
        self._experiment_permissions = experiment_permissions
        self._registered_model_permissions = registered_model_permissions

    @property
    def username(self):
        return self._username

    @property
    def password_hash(self):
        return self._password_hash

    @property
    def is_admin(self):
        return self._is_admin

    @is_admin.setter
    def is_admin(self, is_admin):
        self._is_admin = is_admin

    @property
    def experiment_permissions(self):
        return self._experiment_permissions

    @experiment_permissions.setter
    def experiment_permissions(self, experiment_permissions):
        self._experiment_permissions = experiment_permissions

    @property
    def registered_model_permissions(self):
        return self._registered_model_permissions

    @registered_model_permissions.setter
    def registered_model_permissions(self, registered_model_permissions):
        self._registered_model_permissions = registered_model_permissions


class ExperimentPermission:
    def __init__(
        self,
        experiment_id,
        user_id,
        permission,
    ):
        self._experiment_id = experiment_id
        self._user_id = user_id
        self._permission = permission

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def user_id(self):
        return self._user_id

    @property
    def permission(self):
        return self._permission

    @permission.setter
    def permission(self, permission):
        self._permission = permission


class RegisteredModelPermission:
    def __init__(
        self,
        name,
        user_id,
        permission,
    ):
        self._name = name
        self._user_id = user_id
        self._permission = permission

    @property
    def name(self):
        return self.name

    @property
    def user_id(self):
        return self._user_id

    @property
    def permission(self):
        return self._permission

    @permission.setter
    def permission(self, permission):
        self._permission = permission

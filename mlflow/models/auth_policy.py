from mlflow.models.resources import Resource, _ResourceBuilder
from mlflow.utils.annotations import experimental


@experimental(version="2.21.0")
class UserAuthPolicy:
    """
    A minimal list of scopes that the user should have access to
    in order to invoke this model

    Note: This is only compatible with Databricks Environment currently.
    TODO: Add Databricks Documentation for User Auth Policy

    Args:
        api_scopes: A list of scopes. Example: "vectorsearch.vector-search-indexes", "sql"
    """

    def __init__(self, api_scopes: list[str]):
        self._api_scopes = api_scopes

    @property
    def api_scopes(self) -> list[str]:
        return self._api_scopes

    @api_scopes.setter
    def api_scopes(self, value: list[str]):
        self._api_scopes = value

    def to_dict(self):
        return {"api_scopes": self.api_scopes}


class SystemAuthPolicy:
    """
    System Auth Policy, which defines a list of resources required to
    serve this model
    """

    def __init__(self, resources: list[Resource]):
        self._resources = resources

    @property
    def resources(self) -> list[Resource]:
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        self._resources = value

    def to_dict(self):
        serialized_resources = _ResourceBuilder.from_resources(self.resources)
        return {"resources": serialized_resources}


class AuthPolicy:
    """
    Specifies the authentication policy for the model, which includes two key
    components.
        System Auth Policy: A list of resources required to serve this model
        User Auth Policy: A minimal list of scopes that the user should
                          have access to, in order to invoke this model
    """

    def __init__(
        self,
        user_auth_policy: UserAuthPolicy | None = None,
        system_auth_policy: SystemAuthPolicy | None = None,
    ):
        self.user_auth_policy = user_auth_policy
        self.system_auth_policy = system_auth_policy

    def to_dict(self):
        """
        Serialize Auth Policy to a dictionary
        """
        return {
            "system_auth_policy": self.system_auth_policy.to_dict()
            if self.system_auth_policy
            else {},
            "user_auth_policy": self.user_auth_policy.to_dict() if self.user_auth_policy else {},
        }

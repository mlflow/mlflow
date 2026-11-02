"""Client-side auth provider that forwards the MLflow Assistant's delegation credential.

An Assistant tool subprocess is started with ``MLFLOW_TRACKING_AUTH`` set to this provider's name
and a short-lived credential in ``_MLFLOW_ASSISTANT_DELEGATION_TOKEN``. The MLflow REST client then
attaches the credential on every outgoing request so the server can authenticate the call as the
session owner (see ``mlflow/server/assistant/delegation.py``). This provider only forwards a
pre-minted credential; it never holds the signing key.
"""

from requests.auth import AuthBase

from mlflow.environment_variables import _MLFLOW_ASSISTANT_DELEGATION_TOKEN
from mlflow.tracking.request_auth.abstract_request_auth_provider import RequestAuthProvider

ASSISTANT_DELEGATION_AUTH_NAME = "assistant-delegation"
# Dedicated header so the credential is never mistaken for a real Basic/Bearer credential.
ASSISTANT_DELEGATION_HEADER = "X-MLflow-Assistant-Delegation"


class _AssistantDelegationAuth(AuthBase):
    def __call__(self, request):
        if token := _MLFLOW_ASSISTANT_DELEGATION_TOKEN.get():
            request.headers[ASSISTANT_DELEGATION_HEADER] = token
        return request


class AssistantDelegationRequestAuthProvider(RequestAuthProvider):
    """Attach the Assistant delegation credential from the environment to outgoing requests."""

    def get_name(self) -> str:
        return ASSISTANT_DELEGATION_AUTH_NAME

    def get_auth(self):
        return _AssistantDelegationAuth()

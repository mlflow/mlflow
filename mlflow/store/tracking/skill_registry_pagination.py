"""Query-bound pagination token and validation helpers for the skill registry.

A page token encodes the originating query parameters (filter string, ordering,
and scope) alongside the offset.  On decode the token validates that every
encoded parameter matches the current request, so a token issued for one query
cannot be replayed against a different query.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass

from mlflow.exceptions import MlflowException
from mlflow.store.tracking import (
    SEARCH_SKILL_REGISTRY_MAX_RESULTS_THRESHOLD,
)


@dataclass(frozen=True)
class SkillRegistryPaginationToken:
    """Opaque page token that binds to the query that created it.

    ``query_scope`` ties the token to a specific search surface and, for
    version searches, to the parent identity so a token issued for one
    skill's versions cannot paginate a different skill's results.

    Examples::

        "skills"

        "skill_versions:acme/code-review"
        "agent_plugins"
        "agent_plugin_versions:acme/my-plugin"
    """

    filter_string: str | None
    order_by: list[str] | None
    offset: int
    query_scope: str

    def encode(self) -> str:
        # sort_keys ensures identical query parameters always produce the same token.
        payload = json.dumps(asdict(self), sort_keys=True)
        return base64.b64encode(payload.encode()).decode()

    @classmethod
    def decode(cls, token: str) -> SkillRegistryPaginationToken:
        try:
            payload = json.loads(base64.b64decode(token))
        except Exception:
            raise MlflowException.invalid_parameter_value(
                "Invalid page token: could not decode."
            ) from None
        try:
            token = cls(
                filter_string=payload["filter_string"],
                order_by=payload["order_by"],
                offset=payload["offset"],
                query_scope=payload["query_scope"],
            )
        except (KeyError, TypeError):
            raise MlflowException.invalid_parameter_value(
                "Invalid page token: missing or malformed fields."
            ) from None
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(token.offset, bool) or not isinstance(token.offset, int) or token.offset < 0:
            raise MlflowException.invalid_parameter_value(
                "Invalid page token: offset must be a non-negative integer."
            )
        return token

    def validate(
        self,
        filter_string: str | None,
        order_by: list[str] | None,
        query_scope: str,
    ) -> None:
        # Clients may send "" or [] on one page and omit the value on another.
        # Treat empty values as absent on both sides so the token still matches.
        if (self.filter_string or None) != (filter_string or None):
            raise MlflowException.invalid_parameter_value(
                "Page token was issued for a different filter_string and cannot "
                "be used with the current request."
            )
        if (self.order_by or None) != (order_by or None):
            raise MlflowException.invalid_parameter_value(
                "Page token was issued for a different order_by and cannot "
                "be used with the current request."
            )
        if self.query_scope != query_scope:
            raise MlflowException.invalid_parameter_value(
                "Page token was issued for a different query scope and cannot "
                "be used with the current request."
            )


def validate_max_results(
    max_results: int,
    threshold: int = SEARCH_SKILL_REGISTRY_MAX_RESULTS_THRESHOLD,
) -> None:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise MlflowException.invalid_parameter_value(
            f"max_results must be an integer, got {max_results!r}."
        )
    if max_results < 1:
        raise MlflowException.invalid_parameter_value(
            f"max_results must be at least 1, got {max_results}."
        )
    if max_results > threshold:
        raise MlflowException.invalid_parameter_value(
            f"max_results must be at most {threshold}, got {max_results}."
        )

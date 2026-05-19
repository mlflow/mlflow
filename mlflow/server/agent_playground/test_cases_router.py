"""FastAPI router for agent_playground test-case CRUD.

Mounted at ``/ajax-api/3.0/mlflow/agent-playground``. Endpoints:

- ``GET    /test-cases?experiment_id=...&max_results=&page_token=``
- ``GET    /test-cases/{test_case_id}?experiment_id=...``
- ``PATCH  /test-cases/{test_case_id}?experiment_id=...`` (partial update)
- ``DELETE /test-cases/{test_case_id}?experiment_id=...``
- ``POST   /test-cases/prompt-for-fix?experiment_id=...``

``experiment_id`` lives in the URL query string on every endpoint (not
in any request body) to match the existing CRUD conventions on this
surface and AIP-122 (parent identifiers go in the URL).

No per-route authorization: relies on the global FastAPI permission
middleware (matching ``job_api_router`` at ``mlflow/server/job_api.py``).
v1 is a single-developer playground; per-experiment authorization
checks belong with the multi-user surface and are tracked as a v2
follow-up (see TODO in ``mlflow/server/auth/__init__.py``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, model_validator

from mlflow.agent_playground.test_cases import prompts, store
from mlflow.agent_playground.test_cases.entities import (
    Expectations,
    PersonaSpec,
    TestCaseRow,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchUtils

_logger = logging.getLogger(__name__)

test_cases_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/agent-playground",
    tags=["AgentPlayground"],
)


_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 1000


class ListTestCasesResponse(BaseModel):
    test_cases: list[TestCaseRow]
    next_page_token: str | None = None


class PromptForFixResponse(BaseModel):
    prompt: str


class PatchTestCaseRequest(BaseModel):
    """Partial-update payload.

    Only fields explicitly supplied are applied. Unsupplied fields are
    left untouched. Lineage and ownership fields (``test_case_id``,
    ``experiment_id``, ``source_*``) are not updatable through this
    endpoint and must travel via the URL / query string instead.

    ``expectations`` is the atomic unit: switching between assertion
    and judge strategies sends the full new ``AssertionExpectations``
    or ``JudgeExpectations`` object. The discriminated union on
    ``kind`` makes orphan-payload misconfiguration structurally
    impossible at the wire layer.

    ``persona`` is nullable on the row; passing ``persona=None`` here
    is a no-op (keep existing), and clearing the persona is opt-in via
    ``clear_persona=True``. An empty body ``{}`` is rejected at the
    wire layer (``_reject_empty_patch``) so a no-op PATCH doesn't
    silently trigger the store's delete-then-insert path (which would
    reset ``created_time``/``source`` and expose the read-side race
    documented on ``store.update_case``).
    """

    expectations: Expectations | None = None
    persona: PersonaSpec | None = None
    clear_persona: bool = False
    rationale_summary: str | None = None
    max_turns: int | None = None
    conversation_messages: list[dict[str, Any]] | None = None
    promoted: bool | None = None

    @model_validator(mode="after")
    def _reject_empty_patch(self) -> PatchTestCaseRequest:
        if (
            not any(
                value is not None
                for value in (
                    self.expectations,
                    self.persona,
                    self.rationale_summary,
                    self.max_turns,
                    self.conversation_messages,
                    self.promoted,
                )
            )
            and not self.clear_persona
        ):
            raise ValueError(
                "PatchTestCaseRequest must carry at least one field to update; "
                "an empty payload would silently delete-then-insert the row."
            )
        return self


def _mlflow_exc_to_http(exc: MlflowException) -> HTTPException:
    return HTTPException(status_code=exc.get_http_status_code(), detail=exc.message)


def _not_implemented_to_http(exc: NotImplementedError) -> HTTPException:
    # ``store.delete_case`` / ``store.update_case`` raise
    # ``NotImplementedError`` on the Databricks tracking backend (the
    # underlying ``EvaluationDataset.delete_records`` path). Surface as
    # 501 with a clean message instead of a 500 stack trace; log the
    # original exception so operators have a server-side breadcrumb
    # when correlating the 501 response back to a backend cause.
    _logger.warning("agent_playground router: operation not supported by backend", exc_info=exc)
    return HTTPException(
        status_code=501,
        detail=f"Operation not supported on the current tracking backend: {exc}",
    )


@test_cases_router.get("/test-cases", response_model=ListTestCasesResponse)
def list_test_cases(
    experiment_id: str = Query(..., min_length=1),
    max_results: int = Query(_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    page_token: str | None = Query(None),
) -> ListTestCasesResponse:
    # Pagination is offset-based and not snapshot-consistent: between
    # paged requests, concurrent inserts/deletes can shift offsets,
    # causing duplicates or skips. Acceptable for v1's expected scale
    # (hundreds of cases per experiment).
    try:
        cases = store.list_cases(experiment_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc

    try:
        page, next_token = SearchUtils.paginate(cases, page_token, max_results)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc
    # ``SearchUtils.create_page_token`` always returns ``bytes``; the
    # ``None`` short-circuit is the only other path.
    next_token_str = next_token.decode("utf-8") if next_token else None
    return ListTestCasesResponse(test_cases=page, next_page_token=next_token_str)


@test_cases_router.get("/test-cases/{test_case_id}", response_model=TestCaseRow)
def get_test_case(
    test_case_id: str,
    experiment_id: str = Query(..., min_length=1),
) -> TestCaseRow:
    try:
        case = store.get_case(experiment_id, test_case_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
        )
    return case


@test_cases_router.patch("/test-cases/{test_case_id}", response_model=TestCaseRow)
def patch_test_case(
    test_case_id: str,
    payload: PatchTestCaseRequest,
    experiment_id: str = Query(..., min_length=1),
) -> TestCaseRow:
    try:
        updated = store.update_case(
            experiment_id,
            test_case_id,
            expectations=payload.expectations,
            persona=payload.persona,
            clear_persona=payload.clear_persona,
            conversation_messages=payload.conversation_messages,
            rationale_summary=payload.rationale_summary,
            max_turns=payload.max_turns,
            promoted=payload.promoted,
        )
    except NotImplementedError as exc:
        raise _not_implemented_to_http(exc) from exc
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc

    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
        )
    return updated


@test_cases_router.delete("/test-cases/{test_case_id}")
def delete_test_case(
    test_case_id: str,
    experiment_id: str = Query(..., min_length=1),
) -> Response:
    try:
        deleted = store.delete_case(experiment_id, test_case_id)
    except NotImplementedError as exc:
        raise _not_implemented_to_http(exc) from exc
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
        )
    return Response(status_code=204)


@test_cases_router.post("/test-cases/prompt-for-fix", response_model=PromptForFixResponse)
def prompt_for_fix(
    test_case_id: str = Query(...),
    experiment_id: str = Query(..., min_length=1),
) -> PromptForFixResponse:
    try:
        prompt = prompts.build_fix_prompt(experiment_id, test_case_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc) from exc
    return PromptForFixResponse(prompt=prompt)

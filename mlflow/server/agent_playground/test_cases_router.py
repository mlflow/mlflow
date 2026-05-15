"""FastAPI router for agent_playground test-case CRUD.

Mounted at ``/ajax-api/3.0/mlflow/agent-playground``. Endpoints:

- ``GET  /test-cases?experiment_id=...&max_results=&page_token=``
- ``GET  /test-cases/{test_case_id}?experiment_id=...``
- ``PATCH /test-cases/{test_case_id}`` (partial update)
- ``DELETE /test-cases/{test_case_id}?experiment_id=...`` (hard delete)
- ``POST /test-cases/prompt-for-fix`` (renders the copy-paste fix prompt)

No per-route auth dependency: relies on the global FastAPI permission
middleware (matching ``job_api_router`` at ``mlflow/server/job_api.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ValidationError

from mlflow.agent_playground.test_cases import prompts, pytest_export, store
from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestCaseRow,
    TestSpec,
    TestStrategy,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchUtils

test_cases_router = APIRouter(
    prefix="/ajax-api/3.0/mlflow/agent-playground",
    tags=["AgentPlayground"],
)


_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 1000


class ListTestCasesResponse(BaseModel):
    test_cases: list[TestCaseRow]
    next_page_token: str | None = None


class PromptForFixRequest(BaseModel):
    experiment_id: str
    test_case_id: str


class PromptForFixResponse(BaseModel):
    prompt: str


class PatchTestCaseRequest(BaseModel):
    """Partial update payload.

    Only fields explicitly supplied are applied. Unsupplied fields are
    left untouched. Lineage and ownership fields (``test_case_id``,
    ``experiment_id``, ``source_*``) are not updatable through this
    endpoint and must travel via the path / query params.
    """

    experiment_id: str
    strategy: TestStrategy | None = None
    rationale_summary: str | None = None
    max_turns: int | None = None
    assertion: AssertionSpec | None = None
    judge: JudgeSpec | None = None
    persona: PersonaSpec | None = None
    clear_persona: bool = False
    conversation_messages: list[dict[str, object]] | None = None
    promoted: bool | None = None


def _merge_spec(existing: TestSpec, patch: PatchTestCaseRequest) -> TestSpec:
    # On strategy switch, auto-clear the inactive strategy's payload.
    # The TestSpec validator rejects orphans (an "assertion" spec with a
    # non-None judge payload, and vice versa), and the UI flow that
    # switches strategy expects the leftover payload to drop, not to
    # error.
    next_strategy = patch.strategy if patch.strategy is not None else existing.strategy
    keep_assertion = next_strategy == "assertion"
    keep_judge = next_strategy == "judge"
    next_assertion = patch.assertion if patch.assertion is not None else existing.assertion
    next_judge = patch.judge if patch.judge is not None else existing.judge
    return TestSpec(
        strategy=next_strategy,
        rationale_summary=(
            patch.rationale_summary
            if patch.rationale_summary is not None
            else existing.rationale_summary
        ),
        max_turns=patch.max_turns if patch.max_turns is not None else existing.max_turns,
        assertion=next_assertion if keep_assertion else None,
        judge=next_judge if keep_judge else None,
        persona=(
            None
            if patch.clear_persona
            else (patch.persona if patch.persona is not None else existing.persona)
        ),
    )


def _mlflow_exc_to_http(exc: MlflowException) -> HTTPException:
    return HTTPException(status_code=exc.get_http_status_code(), detail=exc.message)


def _not_implemented_to_http(exc: NotImplementedError) -> HTTPException:
    # ``store.delete_case`` / ``store.update_case`` raise ``NotImplementedError``
    # on the Databricks tracking backend (the underlying
    # ``EvaluationDataset.delete_records`` path). Surface as 501 with a
    # clean message instead of a 500 stack trace.
    return HTTPException(
        status_code=501,
        detail=f"Operation not supported on the current tracking backend: {exc}",
    )


@test_cases_router.get("/test-cases", response_model=ListTestCasesResponse)
def list_test_cases(
    experiment_id: str = Query(...),
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
        raise _mlflow_exc_to_http(exc)

    try:
        page, next_token = SearchUtils.paginate(cases, page_token, max_results)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)
    next_token_str = next_token.decode("utf-8") if isinstance(next_token, bytes) else next_token
    return ListTestCasesResponse(test_cases=page, next_page_token=next_token_str)


@test_cases_router.get("/test-cases/export")
def export_test_cases(
    experiment_id: str = Query(...),
    format: str = Query("pytest"),
) -> Response:
    if format != "pytest":
        raise HTTPException(
            status_code=400, detail=f"Unsupported export format: {format!r}. Supported: 'pytest'."
        )
    try:
        cases = store.list_cases(experiment_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)
    source = pytest_export.render_pytest_suite(experiment_id, cases)
    return Response(
        content=source,
        media_type="text/x-python",
        headers={
            "Content-Disposition": (
                f'attachment; filename="test_agent_playground_{experiment_id}.py"'
            ),
        },
    )


@test_cases_router.get("/test-cases/{test_case_id}", response_model=TestCaseRow)
def get_test_case(test_case_id: str, experiment_id: str = Query(...)) -> TestCaseRow:
    try:
        case = store.get_case(experiment_id, test_case_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
        )
    return case


@test_cases_router.patch("/test-cases/{test_case_id}", response_model=TestCaseRow)
def patch_test_case(test_case_id: str, payload: PatchTestCaseRequest) -> TestCaseRow:
    try:
        existing = store.get_case(payload.experiment_id, test_case_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)

    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Test case {test_case_id!r} not found in experiment {payload.experiment_id!r}"
            ),
        )

    try:
        next_spec = _merge_spec(existing.spec, payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        updated = store.update_case(
            payload.experiment_id,
            test_case_id,
            spec=next_spec,
            conversation_messages=payload.conversation_messages,
            promoted=payload.promoted,
        )
    except NotImplementedError as exc:
        raise _not_implemented_to_http(exc)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)

    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Test case {test_case_id!r} not found in experiment {payload.experiment_id!r}"
            ),
        )
    return updated


@test_cases_router.delete("/test-cases/{test_case_id}")
def delete_test_case(test_case_id: str, experiment_id: str = Query(...)) -> Response:
    try:
        deleted = store.delete_case(experiment_id, test_case_id)
    except NotImplementedError as exc:
        raise _not_implemented_to_http(exc)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
        )
    return Response(status_code=204)


@test_cases_router.post("/test-cases/prompt-for-fix", response_model=PromptForFixResponse)
def prompt_for_fix(payload: PromptForFixRequest) -> PromptForFixResponse:
    try:
        prompt = prompts.build_fix_prompt(payload.experiment_id, payload.test_case_id)
    except MlflowException as exc:
        raise _mlflow_exc_to_http(exc)
    return PromptForFixResponse(prompt=prompt)

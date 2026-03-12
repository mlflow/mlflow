"""
Gateway guardrail execution and API endpoints.

Guardrails are scorers (judges) that run before or after LLM invocation
to validate or mutate requests/responses.
"""

import json
import logging
import os
import threading

from fastapi import APIRouter, HTTPException, Request

from mlflow.entities.gateway_guardrail import GuardrailConfig, GuardrailHook, GuardrailOperation
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

# Default experiment ID for guardrail-registered scorers
_GUARDRAIL_EXPERIMENT_ID = "0"

# Context variable to prevent recursive guardrail execution.
# When a guardrail judge calls an LLM endpoint that itself has guardrails,
# we skip guardrails on the nested call to avoid infinite recursion.
_guardrail_depth = threading.local()

_MAX_GUARDRAIL_DEPTH = 1


def _is_inside_guardrail() -> bool:
    return getattr(_guardrail_depth, "depth", 0) >= _MAX_GUARDRAIL_DEPTH


def _ensure_gateway_uri(request: Request) -> None:
    """Set MLFLOW_GATEWAY_URI from the incoming request if not already set.

    When the server process runs judges with ``gateway:/`` model URIs, LiteLLM
    needs an HTTP base URL to route through the gateway.  The tracking URI
    inside the server process is typically ``sqlite://`` (not HTTP), so we
    derive the gateway URI from the request's own base URL.
    """
    if MLFLOW_GATEWAY_URI.is_set():
        return
    base = str(request.base_url).rstrip("/")
    os.environ[MLFLOW_GATEWAY_URI.name] = base
    _logger.debug("Set %s=%s from request", MLFLOW_GATEWAY_URI.name, base)

guardrail_router = APIRouter(prefix="/gateway/guardrails", tags=["gateway-guardrails"])


# ─── API Endpoints ───────────────────────────────────────────────────────────


def _register_custom_scorer(
    scorer_name: str, prompt: str, response_schema: str | None = None, model: str | None = None
) -> dict:
    """
    Register a custom judge as a scorer via make_judge + scorer store.

    Returns dict with experiment_id and scorer_version for the registered scorer.
    """
    try:
        from mlflow.genai.judges import make_judge
        from mlflow.genai.scorers.registry import _get_scorer_store

        judge = make_judge(name=scorer_name, instructions=prompt, model=model)
        store = _get_scorer_store()
        version = store.register_scorer(
            experiment_id=_GUARDRAIL_EXPERIMENT_ID,
            scorer=judge,
        )
        _logger.info(
            f"Registered custom guardrail scorer '{scorer_name}' "
            f"(experiment_id={_GUARDRAIL_EXPERIMENT_ID}, version={version})"
        )
        return {
            "experiment_id": _GUARDRAIL_EXPERIMENT_ID,
            "scorer_version": version.scorer_version,
        }
    except Exception:
        _logger.exception(f"Failed to register custom scorer '{scorer_name}'")
        raise


@guardrail_router.post("/add")
async def add_guardrail(request: Request):
    body = await request.json()

    scorer_name = body.get("scorer_name")
    if not scorer_name:
        raise HTTPException(status_code=400, detail="Missing required 'scorer_name'")

    hook = body.get("hook", "PRE")
    operation = body.get("operation", "VALIDATION")

    try:
        hook_enum = GuardrailHook(hook)
        operation_enum = GuardrailOperation(operation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

    config = body.get("config", {})

    # If this is a custom judge with a prompt, register it as a scorer
    if config.get("prompt") and not config.get("builtin_scorer") and not config.get("registered_scorer"):
        try:
            reg_info = _register_custom_scorer(
                scorer_name=scorer_name,
                prompt=config["prompt"],
                response_schema=config.get("response_schema"),
                model=config.get("model"),
            )
            config["registered_scorer"] = scorer_name
            config["experiment_id"] = reg_info["experiment_id"]
            config["scorer_version"] = reg_info["scorer_version"]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to register custom scorer: {e}",
            )

    guardrail = store.add_guardrail(
        scorer_name=scorer_name,
        hook=hook_enum,
        operation=operation_enum,
        endpoint_name=body.get("endpoint_name"),
        order=body.get("order", 0),
        config_json=json.dumps(config) if config else None,
    )

    return {
        "guardrail": {
            "guardrail_id": guardrail.guardrail_id,
            "endpoint_name": guardrail.endpoint_name,
            "scorer_name": guardrail.scorer_name,
            "hook": guardrail.hook.value,
            "operation": guardrail.operation.value,
            "order": guardrail.order,
            "enabled": guardrail.enabled,
            "config": config or None,
        }
    }


@guardrail_router.post("/update")
async def update_guardrail(request: Request):
    body = await request.json()
    guardrail_id = body.get("guardrail_id")
    if not guardrail_id:
        raise HTTPException(status_code=400, detail="Missing required 'guardrail_id'")

    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

    kwargs: dict = {}
    if "scorer_name" in body:
        kwargs["scorer_name"] = body["scorer_name"]
    if "hook" in body:
        try:
            kwargs["hook"] = GuardrailHook(body["hook"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if "operation" in body:
        try:
            kwargs["operation"] = GuardrailOperation(body["operation"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if "config" in body:
        kwargs["config_json"] = json.dumps(body["config"]) if body["config"] is not None else None

    guardrail = store.update_guardrail(guardrail_id=guardrail_id, **kwargs)

    return {
        "guardrail": {
            "guardrail_id": guardrail.guardrail_id,
            "endpoint_name": guardrail.endpoint_name,
            "scorer_name": guardrail.scorer_name,
            "hook": guardrail.hook.value,
            "operation": guardrail.operation.value,
            "order": guardrail.order,
            "enabled": guardrail.enabled,
            "config": json.loads(guardrail.config_json) if guardrail.config_json else None,
        }
    }


@guardrail_router.post("/remove")
async def remove_guardrail(request: Request):
    body = await request.json()
    guardrail_id = body.get("guardrail_id")
    if not guardrail_id:
        raise HTTPException(status_code=400, detail="Missing required 'guardrail_id'")

    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

    store.remove_guardrail(guardrail_id)
    return {}


@guardrail_router.post("/reorder")
async def reorder_guardrails(request: Request):
    body = await request.json()
    guardrail_ids = body.get("guardrail_ids")
    if not guardrail_ids or not isinstance(guardrail_ids, list):
        raise HTTPException(status_code=400, detail="Missing required 'guardrail_ids' (list)")

    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

    store.reorder_guardrails(guardrail_ids)
    return {}


@guardrail_router.get("/list")
async def list_guardrails(endpoint_name: str | None = None):
    store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

    guardrails = store.list_guardrails(endpoint_name=endpoint_name)
    return {
        "guardrails": [
            {
                "guardrail_id": g.guardrail_id,
                "endpoint_name": g.endpoint_name,
                "scorer_name": g.scorer_name,
                "hook": g.hook.value,
                "operation": g.operation.value,
                "order": g.order,
                "enabled": g.enabled,
                "config": json.loads(g.config_json) if g.config_json else None,
            }
            for g in guardrails
        ]
    }


@guardrail_router.post("/test")
async def test_guardrail(request: Request):
    """
    Test a guardrail against provided text or a trace from the endpoint's experiment.

    Accepts either:
      - guardrail_id: look up an existing guardrail from the DB
      - scorer_name + hook + operation + config: test inline (before saving)

    Input can be provided as:
      - text: raw text to test against
      - trace_id + experiment_id: fetch input/output from a trace
    """
    _ensure_gateway_uri(request)
    body = await request.json()

    guardrail_id = body.get("guardrail_id")
    scorer_name = None
    hook_value = None
    operation_value = None
    config_json = None

    if guardrail_id:
        # Look up existing guardrail
        store = _get_store()
        if not isinstance(store, SqlAlchemyStore):
            raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")

        guardrails = store.list_guardrails()
        guardrail = next((g for g in guardrails if g.guardrail_id == guardrail_id), None)
        if not guardrail:
            raise HTTPException(status_code=404, detail=f"Guardrail '{guardrail_id}' not found")

        scorer_name = guardrail.scorer_name
        hook_value = guardrail.hook.value
        operation_value = guardrail.operation.value
        config_json = guardrail.config_json
    else:
        # Inline config for testing before saving
        scorer_name = body.get("scorer_name")
        hook_value = body.get("hook", "PRE")
        operation_value = body.get("operation", "VALIDATION")
        config = body.get("config")
        if not scorer_name:
            raise HTTPException(status_code=400, detail="Provide 'guardrail_id' or 'scorer_name' + 'config'")
        config_json = json.dumps(config) if config else None

    # Resolve test text: either from body or from a trace
    text = body.get("text")
    trace_id = body.get("trace_id")
    experiment_id = body.get("experiment_id")

    trace_input = None
    trace_output = None

    if trace_id and experiment_id:
        store = _get_store()
        if not isinstance(store, SqlAlchemyStore):
            raise HTTPException(status_code=500, detail="Guardrails require SqlAlchemyStore")
        try:
            trace_info = store.get_trace_info(trace_id)
            trace_input = trace_info.request_preview or ""
            trace_output = trace_info.response_preview or ""
        except Exception as e:
            _logger.warning(f"Failed to fetch trace {trace_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch trace: {e}")

        # Use input for PRE guardrails, output for POST guardrails
        if hook_value == "PRE":
            text = trace_input
        else:
            text = trace_output

    if not text:
        raise HTTPException(status_code=400, detail="No text to test. Provide 'text' or 'trace_id' + 'experiment_id'.")

    # Run the scorer
    result = _run_scorer(scorer_name, text, config_json, operation_value, hook=hook_value)

    return {
        "result": {
            "score": result.get("score"),
            "rationale": result.get("rationale"),
            "modified_text": result.get("modified_text"),
        },
        "guardrail": {
            "guardrail_id": guardrail_id,
            "scorer_name": scorer_name,
            "hook": hook_value,
            "operation": operation_value,
        },
        "input_text": text,
        "trace_input": trace_input,
        "trace_output": trace_output,
    }


# ─── Guardrail Execution ────────────────────────────────────────────────────


class GuardrailRejection(Exception):
    """Raised when a validation guardrail rejects a request/response."""

    def __init__(self, guardrail_name: str, reason: str):
        self.guardrail_name = guardrail_name
        self.reason = reason
        super().__init__(f"Guardrail '{guardrail_name}' rejected: {reason}")


def _run_scorer(
    scorer_name: str,
    text: str,
    config_json: str | None = None,
    operation: str = "VALIDATION",
    hook: str = "PRE",
) -> dict:
    """
    Run a scorer/judge against the given text.

    Supports:
      - builtin_scorer: GuardrailsScorer subclasses (ToxicLanguage, DetectPII, etc.)
      - registered_scorer: scorers registered via register_scorer() or make_judge()
      - prompt: custom make_judge-style prompts (registered on add)

    Args:
        hook: "PRE" or "POST" — determines whether the text is passed as
              ``inputs`` or ``outputs`` to the judge template.

    Returns:
        dict with keys:
          - "score": "yes"/"no" for validation, or modified text for mutation
          - "rationale": explanation
          - "modified_text": (mutation only) the raw feedback value
    """
    config = json.loads(config_json) if config_json else {}
    is_mutation = operation == "MUTATION"
    use_inputs = hook == "PRE"

    # Track guardrail depth so that any nested LLM calls
    # (e.g. judge → gateway endpoint) skip guardrails and avoid recursion.
    current_depth = getattr(_guardrail_depth, "depth", 0)
    _guardrail_depth.depth = current_depth + 1
    try:
        return _run_scorer_inner(scorer_name, text, config, is_mutation, use_inputs)
    finally:
        _guardrail_depth.depth = current_depth


def _run_scorer_inner(
    scorer_name: str, text: str, config: dict, is_mutation: bool, use_inputs: bool = False
) -> dict:
    # Run a GuardrailsScorer builtin (these are NLP-based, don't use templates,
    # so always pass as outputs — the text is just analyzed directly)
    if "builtin_scorer" in config:
        return _run_builtin_scorer(config["builtin_scorer"], text, is_mutation)

    # Run a registered scorer (including custom judges that were registered on add)
    if "registered_scorer" in config:
        return _run_registered_scorer(
            config["registered_scorer"],
            config.get("experiment_id", _GUARDRAIL_EXPERIMENT_ID),
            config.get("scorer_version"),
            text,
            is_mutation,
            use_inputs,
        )

    # Inline custom judge prompt (for testing before registration)
    if "prompt" in config:
        return _run_inline_judge(
            scorer_name, config["prompt"], text, is_mutation, model=config.get("model"),
            use_inputs=use_inputs,
        )

    # Legacy: keyword-based filtering
    if "keywords" in config:
        keywords = config["keywords"]
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return {
                    "score": "no",
                    "rationale": f"Input contains blocked keyword: '{keyword}'",
                }
        return {"score": "yes", "rationale": "No blocked keywords found"}

    # Default: pass
    return {"score": "yes", "rationale": "Default pass (no config)"}


def _run_builtin_scorer(builtin_name: str, text: str, is_mutation: bool = False) -> dict:
    """Run a GuardrailsScorer builtin (ToxicLanguage, DetectPII, etc.)."""
    try:
        from mlflow.genai.scorers.guardrails import get_scorer

        scorer = get_scorer(builtin_name)
        feedback = scorer(outputs=text)

        raw_value = feedback.value
        if is_mutation and raw_value is not None:
            return {
                "score": "yes",
                "rationale": feedback.rationale or "Mutation applied",
                "modified_text": str(raw_value),
            }

        value = str(raw_value).lower() if raw_value else "yes"
        passed = value == "yes"
        return {
            "score": "yes" if passed else "no",
            "rationale": feedback.rationale or ("Passed" if passed else "Failed"),
        }
    except Exception as e:
        _logger.exception(f"Error running builtin scorer '{builtin_name}'")
        return {"score": "yes", "rationale": f"Error running scorer (pass): {e}"}


def _run_registered_scorer(
    scorer_name: str,
    experiment_id: str,
    scorer_version: int | None,
    text: str,
    is_mutation: bool = False,
    use_inputs: bool = False,
) -> dict:
    """Run a registered scorer loaded from the scorer registry."""
    try:
        from mlflow.genai.scorers.registry import get_scorer as _get_scorer

        scorer = _get_scorer(
            experiment_id=experiment_id,
            name=scorer_name,
            version=scorer_version,
        )
        kwargs = {"inputs": text} if use_inputs else {"outputs": text}
        feedback = scorer(**kwargs)

        raw_value = feedback.value
        if is_mutation and raw_value is not None:
            return {
                "score": "yes",
                "rationale": feedback.rationale or "Mutation applied",
                "modified_text": str(raw_value),
            }

        value = str(raw_value).lower() if raw_value else "yes"
        passed = value == "yes"
        return {
            "score": "yes" if passed else "no",
            "rationale": feedback.rationale or ("Passed" if passed else "Failed"),
        }
    except Exception as e:
        _logger.exception(f"Error running registered scorer '{scorer_name}'")
        return {"score": "yes", "rationale": f"Error running scorer (pass): {e}"}


def _run_inline_judge(
    name: str, prompt: str, text: str, is_mutation: bool = False, model: str | None = None,
    use_inputs: bool = False,
) -> dict:
    """Run a make_judge inline without registering it (for testing before saving)."""
    try:
        from mlflow.genai.judges import make_judge

        judge = make_judge(name=name, instructions=prompt, model=model)
        kwargs = {"inputs": text} if use_inputs else {"outputs": text}
        feedback = judge(**kwargs)

        raw_value = feedback.value
        if is_mutation and raw_value is not None:
            return {
                "score": "yes",
                "rationale": feedback.rationale or "Mutation applied",
                "modified_text": str(raw_value),
            }

        value = str(raw_value).lower() if raw_value else "yes"
        passed = value == "yes"
        return {
            "score": "yes" if passed else "no",
            "rationale": feedback.rationale or ("Passed" if passed else "Failed"),
        }
    except Exception as e:
        _logger.exception(f"Error running inline judge '{name}'")
        return {"score": "yes", "rationale": f"Error running judge (pass): {e}"}


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract user-facing text from chat messages for guardrail evaluation."""
    parts = []
    for msg in messages:
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
        elif isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
    return "\n".join(parts)


def _extract_text_from_response(response: dict) -> str:
    """Extract assistant text from a chat completion response."""
    parts = []
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
    return "\n".join(parts)


def run_pre_guardrails(
    guardrails: list[GuardrailConfig],
    body: dict,
) -> dict:
    """
    Run PRE-invocation guardrails on the request body.

    For VALIDATION guardrails: if scorer returns "no", raise GuardrailRejection.
    For MUTATION guardrails: modify the request body in-place.

    Returns:
        Potentially modified request body.

    Raises:
        GuardrailRejection: If a validation guardrail rejects the request.
    """
    # Skip guardrails if we're already inside a guardrail evaluation
    # (e.g. a judge LLM call routed through a guardrailed endpoint)
    if _is_inside_guardrail():
        _logger.debug("Skipping PRE guardrails: already inside guardrail evaluation")
        return body

    pre_guardrails = sorted(
        [g for g in guardrails if g.hook == GuardrailHook.PRE],
        key=lambda g: g.order,
    )

    if not pre_guardrails:
        return body

    input_text = _extract_text_from_messages(body.get("messages", []))

    for guardrail in pre_guardrails:
        _logger.debug(f"Running PRE guardrail: {guardrail.scorer_name}")
        result = _run_scorer(
            guardrail.scorer_name, input_text, guardrail.config_json, guardrail.operation.value,
            hook="PRE",
        )

        if guardrail.operation == GuardrailOperation.VALIDATION:
            if result.get("score") == "no":
                raise GuardrailRejection(
                    guardrail_name=guardrail.scorer_name,
                    reason=result.get("rationale", "Rejected by guardrail"),
                )
        elif guardrail.operation == GuardrailOperation.MUTATION:
            if "modified_text" in result:
                # Replace the last user message content with the modified text
                for msg in reversed(body.get("messages", [])):
                    if msg.get("role") == "user":
                        msg["content"] = result["modified_text"]
                        break

    return body


def run_post_guardrails(
    guardrails: list[GuardrailConfig],
    response: dict,
) -> dict:
    """
    Run POST-invocation guardrails on the response.

    For VALIDATION guardrails: if scorer returns "no", raise GuardrailRejection.
    For MUTATION guardrails: modify the response.

    Returns:
        Potentially modified response dict.

    Raises:
        GuardrailRejection: If a validation guardrail rejects the response.
    """
    if _is_inside_guardrail():
        _logger.debug("Skipping POST guardrails: already inside guardrail evaluation")
        return response

    post_guardrails = sorted(
        [g for g in guardrails if g.hook == GuardrailHook.POST],
        key=lambda g: g.order,
    )

    if not post_guardrails:
        return response

    output_text = _extract_text_from_response(response)

    for guardrail in post_guardrails:
        _logger.debug(f"Running POST guardrail: {guardrail.scorer_name}")
        result = _run_scorer(
            guardrail.scorer_name, output_text, guardrail.config_json, guardrail.operation.value,
            hook="POST",
        )

        if guardrail.operation == GuardrailOperation.VALIDATION:
            if result.get("score") == "no":
                raise GuardrailRejection(
                    guardrail_name=guardrail.scorer_name,
                    reason=result.get("rationale", "Rejected by guardrail"),
                )
        elif guardrail.operation == GuardrailOperation.MUTATION:
            if "modified_text" in result:
                for choice in response.get("choices", []):
                    if "message" in choice:
                        choice["message"]["content"] = result["modified_text"]

    return response

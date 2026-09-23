import json
import math
import os
import re
from dataclasses import asdict
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, Trace
from mlflow.environment_variables import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.genai.utils.gateway_utils import _resolve_gateway_uri
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, UNAUTHENTICATED
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.annotations import experimental
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.request_utils import _get_http_response_with_retries
from mlflow.utils.rest_utils import http_request

_RETRY_CODES = (429, 500, 502, 503, 504, 529)


@experimental(version="3.17.0")
class JevScorer(Scorer):
    """A serializable scorer using TypeSafe's Jev evaluation models.

    Use :func:`make_jev_scorer` to create a scorer. Direct ``typesafe:/`` models
    read ``TYPESAFE_API_KEY`` at invocation time. Registered scorers require a
    ``gateway:/`` endpoint whose credentials are managed by the tracking server.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    # The tracking store returns None when a registered scorer's endpoint has been deleted.
    model: str | None
    question: str
    answer_type: Literal["noul", "choice", "score"] = "noul"
    criteria: dict[str, str] | list[str] | None = None
    threshold: float | None = Field(default=None, ge=0, le=1, strict=True)

    @field_validator("name", "question")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not re.fullmatch(
            r"(?:typesafe:/[a-zA-Z0-9][a-zA-Z0-9_.-]*|gateway:/[a-zA-Z0-9_.-]+)", value
        ):
            raise ValueError("model must be typesafe:/<model-name> or gateway:/<endpoint-name>")
        return value

    @model_validator(mode="after")
    def _validate_question(self):
        if self.answer_type == "noul":
            if self.criteria is not None and (
                not isinstance(self.criteria, dict)
                or not set(self.criteria).issubset({"true", "false"})
            ):
                raise ValueError(
                    "noul criteria must be a dictionary with 'true' and/or 'false' keys"
                )
        elif self.answer_type == "choice":
            if not isinstance(self.criteria, dict) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("choice criteria must contain between 1 and 255 options")
            if any(not label.strip() for label in self.criteria):
                raise ValueError("choice labels must not be empty")
        elif not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
            raise ValueError("score criteria must contain between 2 and 10 ordered descriptions")
        if self.threshold is not None and self.answer_type != "noul":
            raise ValueError("threshold is only supported for noul questions")
        return self

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.JEV

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return asdict(
            SerializedScorer(
                name=self.name,
                description=self.description,
                aggregations=self.aggregations,
                timeout=self.timeout,
                jev_scorer_pydantic_data=BaseModel.model_dump(self, **kwargs),
            )
        )

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """Evaluate inputs, outputs, and expectations, resolving missing values from a trace."""
        if self.model is None:
            raise MlflowException.invalid_parameter_value(
                "This Jev scorer has no model. Its gateway endpoint may have been deleted. "
                "Configure a new gateway:/ endpoint before evaluating."
            )
        if trace is not None:
            # Trace utilities load evaluation dependencies that are optional in the skinny client.
            from mlflow.genai.utils.trace_utils import (
                resolve_expectations_from_trace,
                resolve_inputs_from_trace,
                resolve_outputs_from_trace,
            )

            inputs = resolve_inputs_from_trace(inputs, trace)
            outputs = resolve_outputs_from_trace(outputs, trace)
            expectations = resolve_expectations_from_trace(expectations, trace)

        question = {"type": self.answer_type, "instructions": self.question}
        if self.criteria is not None:
            question["criteria"] = self.criteria
        # Use MLflow's trace encoding for structured inputs such as Pydantic models and arrays.
        state = json.loads(
            json.dumps(
                {"inputs": inputs, "outputs": outputs, "expectations": expectations},
                cls=TraceJSONEncoder,
                allow_nan=False,
            )
        )
        response = self._invoke({
            "model": self.model.split(":/", 1)[1],
            "state": state,
            "questions": {"evaluation": question},
        })
        value, metadata = self._parse_answer(response)
        return Feedback(
            name=self.name,
            value=value,
            metadata=metadata,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=self.model
            ),
            trace_id=trace.info.trace_id if trace else None,
        )

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.model.startswith("gateway:/"):
                response = http_request(
                    host_creds=get_default_host_creds(_resolve_gateway_uri()),
                    endpoint="/gateway/typesafe/v1/systemone",
                    method="POST",
                    extra_headers={MLFLOW_GATEWAY_CALLER_HEADER: GatewayCaller.JUDGE.value},
                    json=payload,
                    timeout=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get(),
                    retry_codes=_RETRY_CODES,
                    raise_on_status=False,
                    allow_redirects=False,
                )
            else:
                if not (api_key := os.environ.get("TYPESAFE_API_KEY")):
                    raise MlflowException.invalid_parameter_value(
                        "Set TYPESAFE_API_KEY to invoke a typesafe:/ model, or configure a "
                        "TypeSafe gateway endpoint and use gateway:/<endpoint-name>."
                    )
                # Do not forward MLflow tracking credentials or workspace headers to TypeSafe.
                response = _get_http_response_with_retries(
                    method="POST",
                    url="https://api.typesafe.ai/v1/systemone",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    max_retries=3,
                    backoff_factor=1,
                    backoff_jitter=0.1,
                    retry_codes=_RETRY_CODES,
                    timeout=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get(),
                    raise_on_status=False,
                    allow_redirects=False,
                )
        except requests.RequestException:
            raise MlflowException(
                "Failed to connect to the Jev evaluation endpoint.", error_code=INTERNAL_ERROR
            ) from None

        if not 200 <= response.status_code < 300:
            error_code = (
                UNAUTHENTICATED
                if response.status_code in (401, 403)
                else BAD_REQUEST
                if response.status_code < 500
                else INTERNAL_ERROR
            )
            raise MlflowException(
                f"Jev evaluation failed with HTTP {response.status_code}. Check the endpoint "
                "configuration, credentials, and question schema.",
                error_code=error_code,
            )
        try:
            return response.json()
        except ValueError:
            raise MlflowException(
                "Jev evaluation returned invalid JSON.", error_code=BAD_REQUEST
            ) from None

    def _parse_answer(self, response: dict[str, Any]) -> tuple[Any, dict[str, str]]:
        try:
            answer = response["answers"]["evaluation"]
            if answer["type"] != self.answer_type or not isinstance(response["model"], str):
                raise ValueError
            metadata = {"jev.model": response["model"]}
            if self.answer_type == "noul":
                probability = _number(answer["noul"], 0, 1)
                metadata["jev.probability"] = json.dumps(probability)
                value = probability if self.threshold is None else probability >= self.threshold
            else:
                probabilities = answer["probabilities"]
                expected_keys = (
                    set(self.criteria)
                    if self.answer_type == "choice"
                    else {str(i) for i in range(len(self.criteria))}
                )
                if not isinstance(probabilities, dict) or set(probabilities) != expected_keys:
                    raise ValueError
                for probability in probabilities.values():
                    _number(probability, 0, 1)
                if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-3):
                    raise ValueError
                metadata["jev.probabilities"] = json.dumps(probabilities)
                metadata["jev.confidence"] = json.dumps(_number(answer["confidence"], 0, 1))
                if self.answer_type == "choice":
                    value = answer["choice"]
                    if value not in expected_keys:
                        raise ValueError
                else:
                    value = _number(answer["score"], 0, len(self.criteria) - 1)
                    legend = answer["legend"]
                    if (
                        not isinstance(legend, dict)
                        or set(legend) != expected_keys
                        or any(not isinstance(v, str) for v in legend.values())
                    ):
                        raise ValueError
                    metadata["jev.legend"] = json.dumps(legend)
            for field, key in (
                ("input_tokens", AssessmentMetadataKey.JUDGE_INPUT_TOKENS),
                ("output_tokens", AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS),
            ):
                if (count := response.get("usage", {}).get(field)) is not None:
                    if type(count) is not int or count < 0:
                        raise ValueError
                    metadata[key] = str(count)
            return value, metadata
        except (KeyError, TypeError, ValueError, AttributeError):
            raise MlflowException(
                f"Jev evaluation returned an invalid {self.answer_type} answer.",
                error_code=BAD_REQUEST,
            ) from None


def _number(value: Any, minimum: float, maximum: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError
    return float(value)


@experimental(version="3.17.0")
def make_jev_scorer(
    *,
    name: str,
    model: str,
    question: str,
    answer_type: Literal["noul", "choice", "score"] = "noul",
    criteria: dict[str, str] | list[str] | None = None,
    threshold: float | None = None,
) -> JevScorer:
    """Create a scorer using TypeSafe's Jev models.

    Args:
        name: Name of the scorer and its feedback.
        model: ``typesafe:/jev-latest`` for local evaluation using ``TYPESAFE_API_KEY``,
            or ``gateway:/<endpoint-name>`` for an OSS MLflow TypeSafe gateway endpoint.
            Registered and automatic scorers require a gateway endpoint.
        question: Literal evaluation instructions. The model receives a state object with
            ``inputs``, ``outputs``, and ``expectations``; refer to those fields in the question.
        answer_type: ``noul`` returns a yes probability, ``choice`` returns a label,
            and ``score`` returns a probability-weighted, zero-based rubric level.
        criteria: Optional descriptions keyed by ``"true"`` and ``"false"`` for noul;
            a dictionary of 1-255 labels to descriptions for choice; or an ordered
            list of 2-10 descriptions for score.
        threshold: For noul only, return a boolean indicating whether the probability is
            at least this value (0-1). The original probability is retained in metadata.

    Returns:
        A :class:`JevScorer` usable with :func:`mlflow.genai.evaluate`. Feedback metadata
        preserves probabilities, confidence, and rubric legends returned by the model.
        Jev does not produce a text rationale.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import make_jev_scorer

            relevance = make_jev_scorer(
                name="relevance",
                model="typesafe:/jev-latest",
                question="Does outputs answer the user's question in inputs?",
                threshold=0.7,
            )
            feedback = relevance(inputs="What is 2 + 2?", outputs="4")
    """
    if model is None:
        raise MlflowException.invalid_parameter_value("A model is required to create a Jev scorer.")
    return JevScorer(
        name=name,
        model=model,
        question=question,
        answer_type=answer_type,
        criteria=criteria,
        threshold=threshold,
    )

from __future__ import annotations

from typing import Any

from mlflow.types.chat import Function


class FunctionCall(Function):
    arguments: str | dict[str, Any] | None = None
    outputs: Any | None = None
    exception: str | None = None

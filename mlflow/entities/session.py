from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from mlflow.tracing.constant import TraceMetadataKey

if TYPE_CHECKING:
    from mlflow.entities import Trace


class Session:
    """
    A session object representing a group of traces that share the same session ID.

    Sessions typically represent multi-turn conversations or related interactions.
    This class provides convenient access to the session ID and allows iteration
    over the traces in the session.

    Args:
        traces: A list of Trace objects that belong to this session.
    """

    def __init__(self, traces: list[Trace]):
        self._traces = traces

    @property
    def id(self) -> str | None:
        if not self._traces:
            return None
        return self._traces[0].info.request_metadata.get(TraceMetadataKey.TRACE_SESSION)

    @property
    def traces(self) -> list[Trace]:
        return self._traces

    def __iter__(self) -> Iterator[Trace]:
        return iter(self._traces)

    def __len__(self) -> int:
        return len(self._traces)

    def __getitem__(self, index: int) -> Trace:
        return self._traces[index]

    def __repr__(self) -> str:
        return f"Session(id={self.id!r})"

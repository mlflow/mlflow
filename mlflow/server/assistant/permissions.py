"""In-memory broker for session-scoped, interactive tool-call permissions.

When an assistant provider that executes tools in-process (currently the
OpenAI-compatible provider) wants to run a tool while the session is not in
full-access mode, it surfaces a permission request to the user and awaits their
Yes/No answer. The awaiting coroutine (running inside the SSE ``astream``
generator) and the HTTP endpoint that delivers the answer live in the same
server process, so the decision is delivered directly through an
``asyncio.Future`` with no polling.

Pending requests are keyed by MLflow session id and held only in memory, so a
decision never leaks across sessions or survives a restart.
"""

import asyncio


class PermissionBroker:
    def __init__(self) -> None:
        # session_id -> {request_id -> Future[bool]}
        self._pending: dict[str, dict[str, asyncio.Future[bool]]] = {}

    def register(self, session_id: str, request_id: str) -> "asyncio.Future[bool]":
        """Register a pending request and return its future.

        Called synchronously *before* the permission-request event is emitted,
        so the future exists by the time a decision can arrive.
        """
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending.setdefault(session_id, {})[request_id] = future
        return future

    async def wait(self, session_id: str, request_id: str) -> bool:
        """Block until a registered request is resolved, then clean it up."""
        future = self._pending.get(session_id, {}).get(request_id)
        if future is None:
            return False
        try:
            return await future
        finally:
            self._pending.get(session_id, {}).pop(request_id, None)

    async def request(self, session_id: str, request_id: str) -> bool:
        """Register a pending request and block until it is resolved.

        Returns True if the user allowed the tool call, False if denied.
        """
        self.register(session_id, request_id)
        return await self.wait(session_id, request_id)

    def resolve(self, session_id: str, request_id: str, allow: bool) -> None:
        """Deliver a decision for a pending request. No-op if unknown/settled."""
        future = self._pending.get(session_id, {}).get(request_id)
        if future is not None and not future.done():
            future.set_result(allow)

    def deny_all(self, session_id: str) -> None:
        """Resolve every pending request for a session as denied."""
        for future in self._pending.get(session_id, {}).values():
            if not future.done():
                future.set_result(False)

    def clear(self, session_id: str) -> None:
        """Drop all state for a finished session."""
        self.deny_all(session_id)
        self._pending.pop(session_id, None)


permission_broker = PermissionBroker()

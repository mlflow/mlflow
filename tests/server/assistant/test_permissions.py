import asyncio

import pytest

from mlflow.server.assistant.permissions import PermissionBroker


@pytest.fixture
def broker():
    return PermissionBroker()


def test_full_access_defaults_to_false(broker):
    assert broker.is_full_access("session-1") is False


def test_set_and_get_full_access(broker):
    broker.set_full_access("session-1", True)
    assert broker.is_full_access("session-1") is True
    assert broker.is_full_access("session-2") is False
    broker.set_full_access("session-1", False)
    assert broker.is_full_access("session-1") is False


@pytest.mark.parametrize("allow", [True, False])
@pytest.mark.asyncio
async def test_request_resolves_with_decision(broker, allow):
    async def resolve_soon():
        # Wait until the request has registered its future before resolving.
        await asyncio.sleep(0.01)
        broker.resolve("session-1", "req-1", allow)

    resolver = asyncio.create_task(resolve_soon())
    result = await broker.request("session-1", "req-1")
    await resolver
    assert result is allow


@pytest.mark.asyncio
async def test_deny_all_unblocks_pending_requests(broker):
    async def deny_soon():
        await asyncio.sleep(0.01)
        broker.deny_all("session-1")

    denier = asyncio.create_task(deny_soon())
    results = await asyncio.gather(
        broker.request("session-1", "req-1"),
        broker.request("session-1", "req-2"),
    )
    await denier
    assert results == [False, False]


@pytest.mark.asyncio
async def test_resolve_is_idempotent(broker):
    async def resolve_twice():
        await asyncio.sleep(0.01)
        broker.resolve("session-1", "req-1", True)
        # A second resolve must not raise even though the future is settled.
        broker.resolve("session-1", "req-1", False)

    resolver = asyncio.create_task(resolve_twice())
    result = await broker.request("session-1", "req-1")
    await resolver
    assert result is True


def test_resolve_unknown_request_is_noop(broker):
    # Resolving before any request exists should not raise.
    broker.resolve("session-1", "missing", True)


@pytest.mark.asyncio
async def test_request_cleans_up_on_cancellation(broker):
    task = asyncio.create_task(broker.request("session-1", "req-1"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):  # noqa: PT011
        await task
    # The future must be removed so a later resolve is a harmless no-op.
    broker.resolve("session-1", "req-1", True)


@pytest.mark.asyncio
async def test_clear_removes_session_state(broker):
    broker.set_full_access("session-1", True)
    resolver = asyncio.create_task(_resolve_after(broker, "session-1", "req-1", True))
    await broker.request("session-1", "req-1")
    await resolver
    broker.clear("session-1")
    assert broker.is_full_access("session-1") is False


async def _resolve_after(broker, session_id, request_id, allow):
    await asyncio.sleep(0.01)
    broker.resolve(session_id, request_id, allow)

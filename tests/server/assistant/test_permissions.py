import asyncio

import pytest

from mlflow.server.assistant.permissions import PermissionBroker


@pytest.fixture
def broker():
    return PermissionBroker()


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
async def test_clear_drops_pending_requests(broker):
    broker.register("session-1", "req-1")
    broker.clear("session-1")
    # After clearing, resolving the dropped request is a harmless no-op.
    broker.resolve("session-1", "req-1", True)

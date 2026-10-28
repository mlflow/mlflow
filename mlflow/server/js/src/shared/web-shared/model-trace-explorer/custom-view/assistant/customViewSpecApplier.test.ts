import { describe, test, expect, jest } from '@jest/globals';
import {
  registerCustomViewSpecApplier,
  getCustomViewSpecApplier,
  getCurrentApplierSessionId,
  waitForCustomViewSpecApplier,
  type CustomViewSpecApplier,
} from './customViewSpecApplier';

const makeApplier = (): CustomViewSpecApplier => jest.fn(async () => ({ ok: true as const }));

describe('registerCustomViewSpecApplier / getCustomViewSpecApplier', () => {
  test('returns null when nothing is registered', () => {
    expect(getCustomViewSpecApplier()).toBeNull();
    expect(getCurrentApplierSessionId()).toBeUndefined();
  });

  test('registers and reads back the applier and its session id', () => {
    const applier = makeApplier();
    const unregister = registerCustomViewSpecApplier('sess-1', applier);
    expect(getCustomViewSpecApplier()).toBe(applier);
    expect(getCurrentApplierSessionId()).toBe('sess-1');
    unregister();
  });

  test('unregister clears the slot', () => {
    const unregister = registerCustomViewSpecApplier('sess-1', makeApplier());
    unregister();
    expect(getCustomViewSpecApplier()).toBeNull();
    expect(getCurrentApplierSessionId()).toBeUndefined();
  });

  test('a later registration overwrites the earlier one (last-writer-wins)', () => {
    const first = makeApplier();
    const second = makeApplier();
    const unregisterFirst = registerCustomViewSpecApplier('sess-1', first);
    const unregisterSecond = registerCustomViewSpecApplier('sess-2', second);

    expect(getCustomViewSpecApplier()).toBe(second);
    expect(getCurrentApplierSessionId()).toBe('sess-2');

    unregisterFirst();
    expect(getCustomViewSpecApplier()).toBe(second);

    unregisterSecond();
    expect(getCustomViewSpecApplier()).toBeNull();
  });
});

describe('waitForCustomViewSpecApplier', () => {
  test('resolves immediately when a matching host is already registered', async () => {
    const applier = makeApplier();
    const unregister = registerCustomViewSpecApplier('sess-1', applier);

    await expect(waitForCustomViewSpecApplier('sess-1')).resolves.toBe(applier);

    unregister();
  });

  test('resolves immediately with the current applier when no session is expected', async () => {
    const applier = makeApplier();
    const unregister = registerCustomViewSpecApplier('sess-1', applier);

    await expect(waitForCustomViewSpecApplier(undefined)).resolves.toBe(applier);

    unregister();
  });

  test('a registration for a DIFFERENT session does not satisfy an active wait', async () => {
    jest.useFakeTimers();
    const promise = waitForCustomViewSpecApplier('sess-expected', 3000);

    // A different host mounts mid-wait; it must not resolve the waiter.
    const wrongApplier = makeApplier();
    const unregisterWrong = registerCustomViewSpecApplier('sess-other', wrongApplier);

    jest.advanceTimersByTime(3000);
    await expect(promise).resolves.toBeNull();

    unregisterWrong();
    jest.useRealTimers();
  });

  test('a registration for the EXPECTED session resolves the pending wait', async () => {
    const promise = waitForCustomViewSpecApplier('sess-expected', 3000);

    const applier = makeApplier();
    const unregister = registerCustomViewSpecApplier('sess-expected', applier);

    await expect(promise).resolves.toBe(applier);

    unregister();
  });

  test('any next registration resolves the wait when no session was expected', async () => {
    const promise = waitForCustomViewSpecApplier(undefined, 3000);

    const applier = makeApplier();
    const unregister = registerCustomViewSpecApplier('sess-new', applier);

    await expect(promise).resolves.toBe(applier);

    unregister();
  });

  test('resolves to null after the timeout when no matching host ever registers', async () => {
    jest.useFakeTimers();
    const promise = waitForCustomViewSpecApplier('sess-expected', 1000);

    jest.advanceTimersByTime(1000);

    await expect(promise).resolves.toBeNull();
    jest.useRealTimers();
  });
});

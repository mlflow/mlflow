import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { pollUntilDone } from './pollUntilDone';

describe('pollUntilDone', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  test('resolves true on the first attempt when already done', async () => {
    const refetch = jest.fn(async () => ['done']);
    const promise = pollUntilDone({
      refetch,
      isDone: (rows) => rows.includes('done'),
      maxAttempts: 5,
      intervalMs: 100,
    });
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test('keeps polling until isDone returns true', async () => {
    const states = [['x'], ['x'], []];
    let i = 0;
    const refetch = jest.fn(async () => states[i++] ?? []);
    const promise = pollUntilDone({
      refetch,
      isDone: (rows) => rows.length === 0,
      maxAttempts: 5,
      intervalMs: 100,
    });
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('resolves false once max attempts is reached', async () => {
    const refetch = jest.fn(async () => ['still here']);
    const promise = pollUntilDone({
      refetch,
      isDone: () => false,
      maxAttempts: 3,
      intervalMs: 100,
    });
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(false);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('swallows refetch failures and keeps trying', async () => {
    let i = 0;
    const refetch = jest.fn(async () => {
      i += 1;
      if (i < 3) throw new Error('flake');
      return ['done'];
    });
    const promise = pollUntilDone({
      refetch,
      isDone: (rows) => rows.includes('done'),
      maxAttempts: 5,
      intervalMs: 100,
    });
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('resolves false when aborted before the first attempt', async () => {
    const refetch = jest.fn(async () => ['still here']);
    const controller = new AbortController();
    controller.abort();
    const promise = pollUntilDone({
      refetch,
      isDone: () => false,
      maxAttempts: 5,
      intervalMs: 100,
      signal: controller.signal,
    });
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(false);
    expect(refetch).not.toHaveBeenCalled();
  });

  test('stops polling and resolves false when aborted between attempts', async () => {
    const refetch = jest.fn(async () => ['still here']);
    const controller = new AbortController();
    const promise = pollUntilDone({
      refetch,
      isDone: () => false,
      maxAttempts: 10,
      intervalMs: 100,
      signal: controller.signal,
    });
    // Let the first attempt complete, then abort before the next iteration runs.
    await jest.advanceTimersByTimeAsync(150);
    controller.abort();
    await jest.runAllTimersAsync();
    await expect(promise).resolves.toBe(false);
    // The first refetch ran; the abort short-circuits subsequent ones.
    expect(refetch.mock.calls.length).toBeLessThan(10);
  });
});

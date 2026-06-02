import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { pollUntilDone } from './pollUntilDone';

// OSS is on Jest 27, which doesn't have `runAllTimersAsync` /
// `advanceTimersByTimeAsync`. Use real timers with a tiny interval (5ms) so the
// suite stays fast while exercising the real async loop.

describe('pollUntilDone', () => {
  beforeEach(() => {
    jest.useRealTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  test('resolves true on the first attempt when already done', async () => {
    const refetch = jest.fn(async () => ['done']);
    await expect(
      pollUntilDone({
        refetch,
        isDone: (rows: string[]) => rows.includes('done'),
        maxAttempts: 5,
        intervalMs: 5,
      }),
    ).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  test('keeps polling until isDone returns true', async () => {
    const states = [['x'], ['x'], []];
    let i = 0;
    const refetch = jest.fn(async () => states[i++] ?? []);
    await expect(
      pollUntilDone({
        refetch,
        isDone: (rows: string[]) => rows.length === 0,
        maxAttempts: 5,
        intervalMs: 5,
      }),
    ).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('resolves false once max attempts is reached', async () => {
    const refetch = jest.fn(async () => ['still here']);
    await expect(
      pollUntilDone({
        refetch,
        isDone: () => false,
        maxAttempts: 3,
        intervalMs: 5,
      }),
    ).resolves.toBe(false);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('swallows refetch failures and keeps trying', async () => {
    let i = 0;
    const refetch = jest.fn(async () => {
      i += 1;
      if (i < 3) throw new Error('flake');
      return ['done'];
    });
    await expect(
      pollUntilDone({
        refetch,
        isDone: (rows: string[]) => rows.includes('done'),
        maxAttempts: 5,
        intervalMs: 5,
      }),
    ).resolves.toBe(true);
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  test('resolves false when aborted before the first attempt', async () => {
    const refetch = jest.fn(async () => ['still here']);
    const controller = new AbortController();
    controller.abort();
    await expect(
      pollUntilDone({
        refetch,
        isDone: () => false,
        maxAttempts: 5,
        intervalMs: 5,
        signal: controller.signal,
      }),
    ).resolves.toBe(false);
    expect(refetch).not.toHaveBeenCalled();
  });

  test('stops polling and resolves false when aborted between attempts', async () => {
    const refetch = jest.fn(async () => ['still here']);
    const controller = new AbortController();
    const promise = pollUntilDone({
      refetch,
      isDone: () => false,
      maxAttempts: 10,
      intervalMs: 20,
      signal: controller.signal,
    });
    // Wait long enough for one attempt to land, then abort.
    await new Promise((r) => setTimeout(r, 30));
    controller.abort();
    await expect(promise).resolves.toBe(false);
    expect(refetch.mock.calls.length).toBeLessThan(10);
  });
});

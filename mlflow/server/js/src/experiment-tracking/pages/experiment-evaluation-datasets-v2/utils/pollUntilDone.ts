interface PollUntilDoneParams<T> {
  /** Re-run on each tick. Anything thrown is swallowed and treated as "not done yet". */
  refetch: () => Promise<T>;
  /** Called with each refetch result. Return true once the desired state is observed. */
  isDone: (result: T) => boolean;
  /** Cap on attempts. Default 60 — matches V1's UC-deletion polling window. */
  maxAttempts?: number;
  /** Delay between attempts. Default 1000ms. */
  intervalMs?: number;
  /**
   * Optional cancel signal. When aborted, the loop short-circuits at the next boundary and
   * resolves to `false`. Callers should use this to abort polling on unmount so refetch/
   * setState callbacks captured in closures stop firing after the component is gone.
   */
  signal?: AbortSignal;
}

// Resolves after `ms` OR immediately when `signal` aborts, whichever is first. Without the
// abort race, an unmount mid-wait still holds the closure (refetch + setState) alive for the
// remainder of the interval before the next abort check fires.
const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    const timeoutId = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timeoutId);
      resolve();
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });

/**
 * Poll a refetch until `isDone(result)` returns true, or `maxAttempts` is exhausted.
 *
 * UC-backed deletes propagate asynchronously — a successful mutation response only means the
 * delete was accepted, not that subsequent reads will reflect it. We poll the list endpoint
 * up to 60 seconds (matching V1) so the UI doesn't show the just-deleted row.
 *
 * Resolves to `true` when `isDone` returned true, `false` if the attempt cap was reached or
 * the `signal` was aborted.
 */
export const pollUntilDone = async <T>({
  refetch,
  isDone,
  maxAttempts = 60,
  intervalMs = 1000,
  signal,
}: PollUntilDoneParams<T>): Promise<boolean> => {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (signal?.aborted) return false;
    try {
      const result = await refetch();
      if (signal?.aborted) return false;
      if (isDone(result)) {
        return true;
      }
    } catch {
      // Treat refetch failure as "not done yet" — retry until the cap.
    }
    await wait(intervalMs, signal);
  }
  return false;
};

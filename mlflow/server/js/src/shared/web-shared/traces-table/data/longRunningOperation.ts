import type { ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import type { SearchTracesPageResponse } from './useTracesPageQuery';

/** Default cadence between polls of a pending search operation. */
export const POLL_INTERVAL_MS = 2000;
/** Consecutive transient poll failures tolerated before the search gives up. */
export const MAX_POLL_ERRORS = 3;

/**
 * Minimal shape of the `databricks.longrunning.Operation` returned by the async trace-search
 * endpoints (both long-running and progressive). Declared locally rather than importing a
 * generated operation type so this module still compiles in OSS builds. The progressive response
 * also carries a `sql_warehouse_id`, which `toPageResult` simply ignores.
 */
export interface LongRunningOperation {
  /** Operation handle: a SQL statement id (long-running) or an opaque token (progressive). */
  name?: string;
  done?: boolean;
  response?: {
    trace_infos?: ModelTraceInfoV3[];
    next_page_token?: string;
    error_code?: string;
    message?: string;
  };
  error?: { error_code?: string; message?: string };
}

/** Abort-aware sleep; rejects with `AbortError` if `signal` fires before the timeout elapses. */
export const delay = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });

/** Map a completed operation to the page shape, or throw on an operation-level error. */
export const toPageResult = (operation: LongRunningOperation): SearchTracesPageResponse => {
  if (operation.error) {
    throw new Error(operation.error.message ?? 'Trace search failed');
  }
  if (operation.response && 'error_code' in operation.response) {
    throw new Error(operation.response.message ?? 'Trace search failed');
  }
  if (operation.response) {
    return {
      trace_infos: operation.response.trace_infos ?? [],
      next_page_token: operation.response.next_page_token,
    };
  }
  // Completed with no response — an empty result set.
  return { trace_infos: [], next_page_token: undefined };
};

/**
 * Poll `pollOnce(name, signal)` until the operation reports `done`. Generic over the poll transport
 * so both the long-running (GET) and progressive (POST) searches share the loop: the first poll
 * fires immediately, later polls wait out `pollIntervalMs`, and up to `MAX_POLL_ERRORS` consecutive
 * transient failures are tolerated. An `AbortError` propagates immediately without retrying.
 */
export const pollUntilDone = async (
  pollOnce: (name: string, signal: AbortSignal | undefined) => Promise<LongRunningOperation>,
  name: string,
  signal: AbortSignal | undefined,
  pollIntervalMs: number,
): Promise<LongRunningOperation> => {
  let consecutiveErrors = 0;

  for (;;) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }

    try {
      const operation = await pollOnce(name, signal);
      consecutiveErrors = 0;
      if (operation?.done) {
        return operation;
      }
    } catch (error) {
      // A superseded/unmounted query aborts the fetch — propagate immediately, don't retry.
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw error;
      }
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_POLL_ERRORS) {
        throw error;
      }
    }

    // The first poll fired immediately above; wait out the interval before each subsequent one.
    await delay(pollIntervalMs, signal);
  }
};

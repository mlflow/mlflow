import { type LongRunningOperation, POLL_INTERVAL_MS, pollUntilDone, toPageResult } from './longRunningOperation';
import { fetchAPI, getAjaxUrl } from '../../model-trace-explorer/ModelTraceExplorer.request.utils';
import {
  buildSearchTracesPagePayload,
  type SearchTracesPageResponse,
  type TracesQueryIdentity,
} from './useTracesPageQuery';

/**
 * Initiate a long-running search. The request body is identical to the synchronous search — the
 * shared `buildSearchTracesPagePayload` keeps the two in lockstep, mirroring the backend contract
 * that `SearchTracesLongRunningRequest` and `SearchTracesRequest` carry the same parameters.
 */
const initiate = (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
  signal: AbortSignal | undefined,
): Promise<LongRunningOperation> =>
  fetchAPI(
    getAjaxUrl('ajax-api/4.0/mlflow/traces/search-long-running'),
    'POST',
    buildSearchTracesPagePayload(identity, pageToken),
    signal,
  ) as Promise<LongRunningOperation>;

const pollOperation = (name: string, signal: AbortSignal | undefined): Promise<LongRunningOperation> =>
  fetchAPI(
    getAjaxUrl(`ajax-api/4.0/mlflow/traces/search/operations/${name}`),
    'GET',
    undefined,
    signal,
  ) as Promise<LongRunningOperation>;

/**
 * Best-effort cancel of the operation's underlying SQL statement. Fire-and-forget: a superseded or
 * unmounted query no longer needs the result, and cancelling an already-finished statement is a
 * harmless no-op. Errors are swallowed so this never surfaces an unhandled rejection.
 */
const cancelOperation = (statementId: string): void => {
  void fetchAPI(getAjaxUrl(`ajax-api/2.0/sql/statements/${statementId}/cancel`), 'POST').catch(() => {});
};

/**
 * Fetch one page of traces via the async "search traces long running" API: initiate the search, then
 * poll the returned operation until it completes. Returns the same `{ trace_infos, next_page_token }`
 * shape as the synchronous search, so the token cache and numbered pagination are unaffected.
 *
 * Cancellation is driven by `signal` — React Query aborts it when the query is superseded (new search,
 * sort, page, or filter) or the component unmounts. That stops the poll loop and cancels the
 * server-side SQL statement so abandoned searches don't keep running.
 *
 * `pollIntervalMs` is injectable purely so tests can exercise multi-poll behavior without fake timers.
 */
export const fetchTracesLongRunningPage = async (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
  signal?: AbortSignal,
  pollIntervalMs: number = POLL_INTERVAL_MS,
): Promise<SearchTracesPageResponse> => {
  const initiated = await initiate(identity, pageToken, signal);

  // The backend never completes on initiate today, but honor it defensively.
  if (initiated.done) {
    return toPageResult(initiated);
  }
  // A nameless operation means no OTEL tables resolved for the location — an empty result set.
  if (!initiated.name) {
    return { trace_infos: [], next_page_token: undefined };
  }

  const operationName = initiated.name;
  const onAbort = () => cancelOperation(operationName);
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    return toPageResult(await pollUntilDone(pollOperation, operationName, signal, pollIntervalMs));
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
};

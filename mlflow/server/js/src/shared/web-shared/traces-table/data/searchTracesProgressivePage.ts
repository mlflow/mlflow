import { type LongRunningOperation, POLL_INTERVAL_MS, pollUntilDone, toPageResult } from './longRunningOperation';
import type { ModelTraceSearchLocation } from '../../model-trace-explorer/ModelTrace.types';
import { fetchAPI, getAjaxUrl } from '../../model-trace-explorer/ModelTraceExplorer.request.utils';
import { type SearchTracesPageResponse, type TracesQueryIdentity } from './useTracesPageQuery';

/**
 * Progressive trace search — a third transport that returns the same
 * `{ trace_infos, next_page_token }` shape as the synchronous and long-running searches, so the
 * token cache and numbered pagination are unaffected. It is expected to be faster than long-running
 * and works **only for V2 trace tables** (the caller gates it on the location type).
 *
 * Unlike long-running (one completed operation per logical page), progressive returns *partial
 * batches*: this transport loops initiate→poll, accumulating traces until the logical page is full
 * or the search is exhausted, then hands the returned token to the next page. The multi-batch
 * accumulation lives entirely here so the hook keeps its "one query = one logical page" invariant.
 *
 * Abort behavior: there is **no cancel endpoint** for progressive (the operation `name` is an opaque
 * token, not a SQL statement id — the long-running `cancelOperation` must not be reused). Abort is
 * therefore cooperative: the loop stops on `signal`, and the in-flight statement is abandoned
 * server-side rather than actively cancelled.
 */

/** Request body for the progressive search. Diffs vs the shared payload: singular `location`,
 * `page_size` (not `max_results`), and a required `client`. */
interface SearchTracesProgressivePayload {
  location?: ModelTraceSearchLocation;
  filter?: string;
  page_size: number;
  order_by?: string[];
  page_token?: string;
  client: 'UI_WEB';
}

/** Upper bound on initiate→poll rounds per logical page. Guards against a server that returns
 * tokens with zero rows forever; on breach we throw so React Query surfaces an error. */
const MAX_PROGRESSIVE_ROUNDS = 50;

/**
 * Build the progressive request body. Not shared with `buildSearchTracesPagePayload`, which sends
 * `locations[]` + `max_results`; progressive takes a singular `location` + `page_size` + `client`.
 */
const buildProgressiveSearchPayload = (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
): SearchTracesProgressivePayload => {
  const payload: SearchTracesProgressivePayload = {
    location: identity.locations[0],
    filter: identity.filter,
    page_size: identity.pageSize,
    order_by: identity.orderBy,
    client: 'UI_WEB',
  };
  if (pageToken) {
    payload.page_token = pageToken;
  }
  return payload;
};

const initiateProgressive = (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
  signal: AbortSignal | undefined,
): Promise<LongRunningOperation> =>
  fetchAPI(
    getAjaxUrl('ajax-api/4.0/mlflow/traces/search-progressive'),
    'POST',
    buildProgressiveSearchPayload(identity, pageToken),
    signal,
  ) as Promise<LongRunningOperation>;

const pollProgressiveOnce = (name: string, signal: AbortSignal | undefined): Promise<LongRunningOperation> =>
  fetchAPI(
    getAjaxUrl('ajax-api/4.0/mlflow/traces/search-progressive/operations'),
    'POST',
    { name },
    signal,
  ) as Promise<LongRunningOperation>;

/**
 * Fetch one logical page of traces via the progressive search API. See the module doc-comment for
 * the transport contract and abort tradeoff.
 *
 * `pollIntervalMs` is injectable purely so tests can exercise multi-poll behavior without fake timers.
 */
export const fetchTracesProgressivePage = async (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
  signal?: AbortSignal,
  pollIntervalMs: number = POLL_INTERVAL_MS,
): Promise<SearchTracesPageResponse> => {
  const accumulated: NonNullable<SearchTracesPageResponse['trace_infos']> = [];
  let cursor = pageToken;
  let rounds = 0;

  for (;;) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
    if (++rounds > MAX_PROGRESSIVE_ROUNDS) {
      throw new Error('Progressive trace search exceeded the maximum number of rounds');
    }

    // Resend the full payload every round (initiate and continuations): the backend re-validates an
    // identity hash + warehouse against the token and reads `page_size` from the request on
    // `NEXT_PAGE` continuations. Do NOT "optimize" this to send only `page_token`.
    let operation = await initiateProgressive(identity, cursor, signal);
    if (!operation.done) {
      // Done-on-initiate is load-bearing (initiate legitimately completes for an empty result), so
      // only poll a pending operation. A nameless pending operation means no result to poll for.
      if (!operation.name) {
        return { trace_infos: accumulated, next_page_token: undefined };
      }
      operation = await pollUntilDone(pollProgressiveOnce, operation.name, signal, pollIntervalMs);
    }

    const batch = toPageResult(operation);
    accumulated.push(...(batch.trace_infos ?? []));
    // Normalize '' → undefined so an exhausted search is unambiguous.
    const nextToken = batch.next_page_token || undefined;

    if (nextToken === undefined) {
      // Search exhausted → this is the last (possibly short) page.
      return { trace_infos: accumulated, next_page_token: undefined };
    }
    if (accumulated.length >= identity.pageSize) {
      // Page full → hand the token to the next logical page.
      return { trace_infos: accumulated, next_page_token: nextToken };
    }
    // Not full and a token remains → keep filling this page.
    cursor = nextToken;
  }
};

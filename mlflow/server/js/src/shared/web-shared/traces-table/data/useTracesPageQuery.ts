import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useQuery, useQueryClient } from '../../query-client/queryClient';
import { SEARCH_MLFLOW_TRACES_QUERY_KEY } from '../../genai-traces-table/hooks/useMlflowTraces';
import { shouldUseLongRunningTracesAPI } from '../../genai-traces-table/utils/FeatureUtils';
import type { ModelTraceInfoV3, ModelTraceSearchLocation } from '../../model-trace-explorer/ModelTrace.types';
import { fetchAPI, getAjaxUrl } from '../../model-trace-explorer/ModelTraceExplorer.request.utils';
import { type TraceTokenCache } from './useTraceTokenCache';
import { fetchTracesLongRunningPage } from './searchTracesLongRunningPage';
import { fetchTracesProgressivePage } from './searchTracesProgressivePage';

/**
 * Normalized response shape consumed by the table: `trace_infos` + a `next_page_token` cursor.
 *
 * OSS MLflow serves the synchronous `ajax-api/3.0/mlflow/traces/search` endpoint (the 4.0 /
 * long-running / progressive variants are Databricks-only and unregistered in OSS). The 3.0 handler
 * returns the trace list under `traces`, so `fetchTracesPage` remaps `traces` → `trace_infos` at the
 * transport seam; the presentational layer only ever sees `trace_infos`.
 */
export interface SearchTracesPageResponse {
  trace_infos?: ModelTraceInfoV3[];
  next_page_token?: string;
}

/** Raw shape returned by OSS `ajax-api/3.0/mlflow/traces/search` (list is under `traces`). */
interface SearchTracesV3RawResponse {
  traces?: ModelTraceInfoV3[];
  next_page_token?: string;
}

export interface SearchTracesPagePayload {
  locations?: ModelTraceSearchLocation[];
  filter?: string;
  max_results: number;
  order_by?: string[];
  page_token?: string;
}

/** Stable identity of a query — everything except the page cursor. Drives the token-cache key. */
export interface TracesQueryIdentity {
  locations: ModelTraceSearchLocation[];
  /** An opaque, consumer-compiled server filter string. The hook never parses or builds it. */
  filter?: string;
  orderBy?: string[];
  /** Optional warehouse id the hook forwards verbatim; it doesn't know what a warehouse is. */
  sqlWarehouseId?: string;
  pageSize: number;
}

/**
 * Deterministic stringify used for both the token-cache key and the React Query key. Keys are
 * enumerated explicitly (not `JSON.stringify(obj)`) so key order can't perturb the string.
 */
const stableQueryKey = (identity: TracesQueryIdentity): string =>
  JSON.stringify([
    identity.locations,
    identity.filter ?? null,
    identity.orderBy ?? null,
    identity.sqlWarehouseId ?? null,
    identity.pageSize,
  ]);

/** Build the shared request payload (used by both the synchronous and long-running transports). */
export const buildSearchTracesPagePayload = (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
): SearchTracesPagePayload => {
  const payload: SearchTracesPagePayload = {
    locations: identity.locations,
    filter: identity.filter,
    max_results: identity.pageSize,
    order_by: identity.orderBy,
  };
  if (pageToken) {
    payload.page_token = pageToken;
  }
  return payload;
};

const fetchTracesPage = async (
  identity: TracesQueryIdentity,
  pageToken: string | undefined,
  signal: AbortSignal | undefined,
): Promise<SearchTracesPageResponse> => {
  const raw = (await fetchAPI(
    getAjaxUrl('ajax-api/3.0/mlflow/traces/search'),
    'POST',
    buildSearchTracesPagePayload(identity, pageToken),
    signal,
  )) as SearchTracesV3RawResponse;
  return { trace_infos: raw.traces, next_page_token: raw.next_page_token };
};

export interface UseTracesPageQueryParams {
  identity: TracesQueryIdentity;
  /** 1-based page. */
  pageIndex: number;
  tokenCache: TraceTokenCache;
  /** When false, the query is disabled (e.g. no SQL warehouse selected yet). Consumer-owned. */
  enabled: boolean;
  /** Called to move to a page (the consumer owns the page-index state, usually via the URL). */
  onPageIndexChange: (next: number) => void;
  /**
   * Opt into the progressive transport. Passed in — not derived from `identity.locations` — so the
   * schema-versioning eligibility decision stays with the product consumer and this fetch mechanism
   * stays product-agnostic (see `data/CLAUDE.md`). Overrides `shouldUseLongRunningTracesAPI`.
   */
  useProgressiveSearch?: boolean;
}

export interface TracesPageQueryResult {
  traces: ModelTraceInfoV3[];
  isLoading: boolean;
  isFetching: boolean;
  /**
   * True while `keepPreviousData` is serving the *prior* query identity's rows — i.e. the displayed
   * `traces` don't belong to the current `identity` yet. Flips in lockstep with the identity (unlike
   * `isFetching`, which lags by a tick), so consumers deriving state from the current filter can tell
   * whether `traces` are stale.
   */
  isPreviousData: boolean;
  error: unknown;
  refetch: () => void;
  /** ms-since-epoch of the last successful resolution, or 0 if never. Drives a refresh label. */
  dataUpdatedAt: number;
  hasNext: boolean;
  hasPrev: boolean;
  /**
   * Navigate to `target` (1-based). With ±1 prev/next nav the target is always adjacent to the
   * resolved current page, so its cursor is already known — no intermediate walk is needed.
   */
  goToPage: (target: number) => void;
}

/**
 * Opt-in, cursor-backed paginated trace query built on web-shared's own transport. It fetches one
 * page at a time and reads `next_page_token` (which the shared `searchTracesV4` throws away),
 * recording it into `tokenCache` so prev/next navigation works. The transport is chosen by
 * `useProgressiveSearch` (the initiate→poll→accumulate `fetchTracesProgressivePage`) then by
 * `shouldUseLongRunningTracesAPI` — the async initiate→poll `fetchTracesLongRunningPage` (immune to
 * the sync search's ~60s timeout) or the synchronous `fetchTracesPage` — but all three return the
 * same shape, so the cursor bookkeeping below is transport-agnostic.
 *
 * The presentational layer imports nothing from here: this is the reusable *fetch mechanism*, not a
 * controller. It never compiles filters, owns URL state, or knows product concepts — `filter` is an
 * opaque consumer-built string, `sqlWarehouseId` is forwarded verbatim, and `enabled` is
 * consumer-owned.
 *
 * Query key reuses `SEARCH_MLFLOW_TRACES_QUERY_KEY` (so the shared refresh/invalidation path also
 * refetches this) with a `'paged'` discriminator. `keepPreviousData` keeps the prior page's rows
 * mounted during a transition (no skeleton flash).
 */
export const useTracesPageQuery = ({
  identity,
  pageIndex,
  tokenCache,
  enabled,
  onPageIndexChange,
  useProgressiveSearch,
}: UseTracesPageQueryParams): TracesPageQueryResult => {
  // Progressive when the consumer opts in; else async initiate→poll when the long-running flag is on;
  // else the synchronous POST. All three return the same `{ trace_infos, next_page_token }` shape, so
  // the token cache and prev/next navigation are transport-agnostic. Memoized on `useProgressiveSearch`
  // (not `[]`) because it changes with the location, not once per session like the long-running flag.
  const fetchPage = useMemo(
    () =>
      useProgressiveSearch
        ? fetchTracesProgressivePage
        : shouldUseLongRunningTracesAPI()
          ? fetchTracesLongRunningPage
          : fetchTracesPage,
    [useProgressiveSearch],
  );

  const queryClient = useQueryClient();

  const cacheKey = useMemo(() => stableQueryKey(identity), [identity]);
  // Reset the token stack whenever the query identity changes — a cursor from a different
  // filter/sort/pageSize points into a different result set. Done during render (pure, idempotent)
  // so the token read below already sees the reset stack on the first render after a change.
  tokenCache.resetIfKeyChanged(cacheKey);

  const pageToken = tokenCache.getTokenForPage(pageIndex);

  const queryKey = useMemo(
    () => [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'paged', cacheKey, pageIndex] as const,
    [cacheKey, pageIndex],
  );

  // Explicitly cancel the *previous* query as soon as the key changes (new search/filter/sort/page).
  // React Query already aborts a superseded query when it removes the old observer, which cancels the
  // fetch and — via the long-running transport's abort handler — the server-side SQL statement; this
  // is a belt-and-suspenders that makes the intent explicit and testable. `exact: true` is load-bearing:
  // `cancelQueries` prefix-matches by default, so a broad key would also abort the newly-mounting query.
  const prevKeyRef = useRef(queryKey);
  useEffect(() => {
    if (prevKeyRef.current !== queryKey) {
      void queryClient.cancelQueries({ queryKey: prevKeyRef.current, exact: true });
      prevKeyRef.current = queryKey;
    }
  }, [queryKey, queryClient]);

  const query = useQuery<SearchTracesPageResponse, Error>({
    queryKey,
    queryFn: ({ signal }) => fetchPage(identity, pageToken, signal),
    enabled,
    keepPreviousData: true,
    refetchOnWindowFocus: false,
    // Finite, non-zero stale time: server-side-filtered pages don't need to re-fetch the instant
    // the user pages back to a cached page (First/Prev must be instant), but they shouldn't be
    // cached forever either — an explicit refresh (which bumps the shared query cache) still
    // refetches. 30s balances snappy back-navigation against staleness.
    staleTime: 30_000,
    cacheTime: 5 * 60_000,
  });

  // Record the resolved page's next-token so the following page becomes reachable. Reading
  // `query.data` in render (rather than an effect) keeps the stack in sync before the pagination
  // bar computes `hasNext` for the current render. The load-bearing guard is `!query.isPreviousData`:
  // in React Query v4, `keepPreviousData` keeps `isSuccess` true while retaining the *prior* page's
  // rows during a page transition, and flags that retained data via `isPreviousData`. Recording on
  // `isSuccess` alone would attribute the previous page's `next_page_token` to the new `pageIndex`,
  // corrupting the cursor cache (a rapid double-Next would then fetch the wrong cursor → duplicate
  // page). We only record once the fetch for *this* page index has actually resolved.
  if (query.isSuccess && !query.isPreviousData && query.data) {
    // A page returning fewer rows than the page size is the last page, regardless of the token the
    // server sent. The long-running search handler returns a real token on *every* non-empty page —
    // even a partial final one — so a short page must be treated as terminal here or Next would never
    // disable. (Progressive already returns `undefined` when exhausted, so the "regardless of token"
    // clause is redundant-but-correct for it.)
    const rowCount = query.data.trace_infos?.length ?? 0;
    const isLastPage = rowCount < identity.pageSize;
    tokenCache.recordNextToken(pageIndex, isLastPage ? null : query.data.next_page_token);
  }

  const traces = useMemo(() => query.data?.trace_infos ?? [], [query.data]);

  // Prev/Next only: the target is always adjacent to the resolved current page, so its cursor is
  // already recorded (page 1 needs none; page N+1's token came from page N's response). No walk.
  const goToPage = useCallback(
    (target: number) => {
      if (target < 1 || target === pageIndex) {
        return;
      }
      onPageIndexChange(target);
    },
    [pageIndex, onPageIndexChange],
  );

  return {
    traces,
    isLoading: query.isLoading && enabled,
    isFetching: query.isFetching,
    isPreviousData: query.isPreviousData,
    error: query.error,
    refetch: query.refetch,
    dataUpdatedAt: query.dataUpdatedAt,
    hasNext: tokenCache.hasNext(pageIndex),
    hasPrev: tokenCache.hasPrev(pageIndex),
    goToPage,
  };
};

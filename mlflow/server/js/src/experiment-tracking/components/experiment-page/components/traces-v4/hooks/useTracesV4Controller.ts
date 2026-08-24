import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { SEARCH_MLFLOW_TRACES_QUERY_KEY, shouldEnableSessionGrouping } from '@databricks/web-shared/genai-traces-table';
import { SESSION_ID_METADATA_KEY, type ModelTraceSearchLocation } from '@databricks/web-shared/model-trace-explorer';
import {
  EMPTY_FILTER_MODEL,
  countActiveFilters,
  useBulkTraceSelection,
  useTracesPageQuery,
  useTraceTokenCache,
  type FilterClause,
  type TraceFilterModel,
  type TracesQueryIdentity,
} from '@databricks/web-shared/traces-table';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
// Reuse the generic (branding-free) datasets-v2 helpers.
import { useDebouncedSearchInput } from './useDebouncedSearchInput';
import { useTracesV4UrlState } from './useTracesV4UrlState';
import { useTracesV4TimeRange } from './useTracesV4TimeRange';
import { useTracesV4Columns } from './useTracesV4Columns';
import { useTracesV4AssessmentColumns } from './useTracesV4AssessmentColumns';
import { useTracesV4ColumnSizing } from './useTracesV4ColumnSizing';
import { useTracesV4TraceCount } from './useTracesV4TraceCount';
import { buildFilter, buildOrderBy } from '../utils/buildTracesV4SearchParams';
import { compileFilterModel, compileTagFilters } from '../utils/filterModel';
import { SEARCH_DEBOUNCE_MS } from '../utils/constants';

// Grouped mode fetches sessions in one big page instead of paginating, so a session isn't split
// across page boundaries. Mirrors the legacy Sessions tab's bounded single fetch. Capped at the
// OSS `SearchTracesV3` server limit (max_results <= 500 in mlflow/server/handlers.py); the managed
// backend allows more, but the shared code must stay within the stricter OSS ceiling.
export const GROUPED_TRACES_LIMIT = 500;

interface UseTracesV4ControllerParams {
  experimentId: string;
}

export interface UseTracesV4ControllerResult {
  url: ReturnType<typeof useTracesV4UrlState>;
  page: ReturnType<typeof useTracesPageQuery>;
  columns: ReturnType<typeof useTracesV4Columns>;
  assessments: ReturnType<typeof useTracesV4AssessmentColumns>;
  columnSizing: ReturnType<typeof useTracesV4ColumnSizing>;
  /** "{n} of {total}" footer count — current page rows out of the experiment total. */
  traceCount: ReturnType<typeof useTracesV4TraceCount>;
  bulk: ReturnType<typeof useBulkTraceSelection>;
  searchInput: ReturnType<typeof useDebouncedSearchInput>;
  filterModel: TraceFilterModel;
  setFilterModel: (next: TraceFilterModel) => void;
  activeFilterCount: number;
  /** ISO time range currently applied (drives the search filter + refresh label context). */
  timeRange: { startTime?: string; endTime?: string };
  /** Navigate to a page, committing any pending typed search first. */
  goToPage: (target: number) => void;
  /** Toggle a URL-persisted click-to-filter tag constraint (from a tag pill in the table). */
  onFilterByTag: (key: string, value: string) => void;
  /** Effective grouped state — the URL flag gated by the session-grouping feature flag. */
  isGroupedBySession: boolean;
  flags: {
    hasActiveSearch: boolean;
    hasNoTracesAtAll: boolean;
    hasNoSearchResults: boolean;
    /** An empty page reached beyond page 1 (paged past the last full page) — "No more results". */
    isEmptyPageBeyondFirst: boolean;
  };
}

/**
 * Orchestrates the traces-v4 data layer: URL state, the shared paginated server query + token cache,
 * column persistence, bulk selection, debounced search, and the filter model. Owned here (not in
 * the page component) so the data lifecycle has a single testable seam and the page stays a thin
 * render layer. Compiling the filter model into a server clause string and the warehouse `enabled`
 * check stay MLflow-side; they feed the shared `useTracesPageQuery` via `identity`.
 */
export const useTracesV4Controller = ({ experimentId }: UseTracesV4ControllerParams): UseTracesV4ControllerResult => {
  const url = useTracesV4UrlState();
  const { timeRangeMs: timeRange, setTimeRange } = useTracesV4TimeRange(experimentId);
  const queryClient = useQueryClient();
  const monitoringConfig = useMonitoringConfig();

  const isGroupedBySession = shouldEnableSessionGrouping() && url.isGroupedBySession;

  const [filterModel, setFilterModel] = useState<TraceFilterModel>(EMPTY_FILTER_MODEL);

  const searchInput = useDebouncedSearchInput({
    committedValue: url.search,
    onCommit: url.setSearch,
    debounceMs: SEARCH_DEBOUNCE_MS,
    // The search box commits only on Enter (submit) and on clear, not on every keystroke.
    commitOnChange: false,
  });

  // Commit any pending typed search before a page transition, so typing then immediately paginating
  // doesn't drop the typed value or let a late commit reset the page back to 1.
  const { setPageIndex: rawSetPageIndex } = url;
  const { submit: submitSearch } = searchInput;

  // OSS traces live under an MLflow experiment (the `SearchTracesV3` handler reads only
  // `location.mlflow_experiment.experiment_id` and ignores UC-schema locations), so we always
  // search the experiment location. The Databricks build instead searches a UC schema.
  const locations = useMemo<ModelTraceSearchLocation[]>(
    () => [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
    [experimentId],
  );

  const filter = useMemo(
    () =>
      buildFilter({
        searchQuery: url.search,
        timeRange,
        // Popover clauses and URL-backed tag clauses are ANDed together (tag filters are their own
        // URL-persisted concept, kept out of the in-memory popover model).
        extraClauses: [...compileFilterModel(filterModel), ...compileTagFilters(url.tagFilters)],
      }),
    [url.search, timeRange, filterModel, url.tagFilters],
  );

  const orderBy = useMemo(() => buildOrderBy(url.sort, url.dir), [url.sort, url.dir]);

  const queryPageSize = isGroupedBySession ? GROUPED_TRACES_LIMIT : url.pageSize;
  const identity = useMemo<TracesQueryIdentity>(
    () => ({ locations, filter, orderBy, pageSize: queryPageSize }),
    [locations, filter, orderBy, queryPageSize],
  );

  const tokenCache = useTraceTokenCache();

  const page = useTracesPageQuery({
    identity,
    pageIndex: url.pageIndex,
    tokenCache,
    // Always enabled in OSS (the sync 3.0 endpoint needs no warehouse). Progressive search is a
    // Databricks-only transport (UC V2 tables), so it's left off.
    enabled: true,
    onPageIndexChange: rawSetPageIndex,
  });

  // Reload button → `monitoringConfig.refresh()` bumps `lastRefreshTime`. For a *relative* range that
  // also moves the time bounds (so the query key changes and refetches on its own), but a *custom*
  // fixed range's bounds are identity-stable — its query key doesn't change, so nothing would refetch.
  // Invalidate the row query on every refresh so the reload always refetches (and, via `isFetching`,
  // shows the skeleton) regardless of range kind. Skip the initial mount so first load isn't
  // double-fetched.
  const lastRefreshTime = monitoringConfig.lastRefreshTime;
  const initialRefreshTimeRef = useRef(lastRefreshTime);
  useEffect(() => {
    if (lastRefreshTime === initialRefreshTimeRef.current) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
  }, [lastRefreshTime, queryClient]);

  const { goToPage: pageGoToPage } = page;
  const goToPage = useCallback(
    (target: number) => {
      submitSearch();
      pageGoToPage(target);
    },
    [submitSearch, pageGoToPage],
  );

  // Recover a stale `?page=N` deep-link: the token cache is memory-only, so on a fresh load it can't
  // walk to page N's cursor. When we're settled on a deep page whose cursor is unknown, reset to 1.
  // Read at render time (like `page.hasNext`) into a primitive so the effect keys off a stable bool
  // rather than the per-render token-cache wrapper.
  const isCurrentPageKnown = tokenCache.isPageKnown(url.pageIndex);
  useEffect(() => {
    if (url.pageIndex > 1 && !page.isLoading && !isCurrentPageKnown) {
      rawSetPageIndex(1);
    }
  }, [url.pageIndex, page.isLoading, isCurrentPageKnown, rawSetPageIndex]);

  // Session column defaults on only when the current page carries session-tagged traces (v1-faithful).
  const hasSessionOnPage = useMemo(
    () => page.traces.some((trace) => Boolean(trace.trace_metadata?.[SESSION_ID_METADATA_KEY])),
    [page.traces],
  );
  const columns = useTracesV4Columns(experimentId, { hasSessionOnPage });
  const assessments = useTracesV4AssessmentColumns(experimentId, page.traces);
  const columnSizing = useTracesV4ColumnSizing(experimentId);
  const traceCount = useTracesV4TraceCount(experimentId, page.traces.length, timeRange);

  const bulk = useBulkTraceSelection(page.traces);

  // Clear selection when the filter set changes (search / time / filter model / sort). Selected
  // traces that no longer match wouldn't be meaningfully selectable, and a stale carry-over would
  // invite accidental deletion. Selection intentionally persists across pagination and page-size.
  const { clear: clearBulk } = bulk;
  useEffect(() => {
    clearBulk();
  }, [url.search, url.sort, url.dir, filterModel, url.tagFilters, timeRange.startTime, timeRange.endTime, clearBulk]);

  // Tag filters count toward the active-filter total alongside the popover clauses, so the empty-state
  // logic distinguishes "no traces at all" from "filters matched nothing" when only a tag is applied.
  const activeFilterCount = countActiveFilters(filterModel) + url.tagFilters.length;
  const hasActiveSearch = url.search.trim().length > 0;
  const hasActiveFilters = hasActiveSearch || activeFilterCount > 0;
  const isSettled = !page.isLoading && !page.isFetching;
  const hasNoResults = isSettled && page.traces.length === 0;
  // An empty result on a page past the first means the user paged one step beyond the last full page
  // (a cursor API can't know page N+1 is empty until it asks). This is distinct from "no traces at
  // all" / "no filter match" — it keeps the pagination bar so the user can step back — so it takes
  // precedence: the initial-empty states are gated to page 1.
  const isEmptyPageBeyondFirst = hasNoResults && url.pageIndex > 1;
  const hasNoTracesAtAll = hasNoResults && !hasActiveFilters && !isEmptyPageBeyondFirst;
  const hasNoSearchResults = hasNoResults && hasActiveFilters && !isEmptyPageBeyondFirst;

  return {
    url,
    page,
    columns,
    assessments,
    columnSizing,
    traceCount,
    bulk,
    searchInput,
    filterModel,
    setFilterModel,
    activeFilterCount,
    timeRange,
    goToPage,
    onFilterByTag: url.addTagFilter,
    isGroupedBySession,
    flags: { hasActiveSearch, hasNoTracesAtAll, hasNoSearchResults, isEmptyPageBeyondFirst },
  };
};

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import {
  createTraceLocationForExperiment,
  doesTraceSupportV4API,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
  searchMlflowTracesQueryFn,
  shouldEnableSessionGrouping,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { filterTracesByAssessmentSourceRunId } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import {
  createTraceV4LongIdentifier,
  SESSION_ID_METADATA_KEY,
  type ModelTraceInfoV3,
  type ModelTraceSearchLocation,
} from '@databricks/web-shared/model-trace-explorer';
import {
  countActiveFilters,
  EMPTY_FILTER_MODEL,
  useBulkTraceSelection,
  useTraceTokenCache,
  useTracesPageQuery,
  type TraceColumnId,
  type TraceFilterModel,
  type TracesPageQueryResult,
  type TracesQueryIdentity,
} from '@databricks/web-shared/traces-table';
import { useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { useDebouncedSearchInput } from '../../components/experiment-page/components/traces-v4/hooks/useDebouncedSearchInput';
import { useTracesV4AssessmentColumns } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4AssessmentColumns';
import { useTracesV4Columns } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Columns';
import { useTracesV4ColumnSizing } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4ColumnSizing';
import { GROUPED_TRACES_LIMIT } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Controller';
import {
  buildFilter,
  buildOrderBy,
  escapeFilterValue,
  isExactTraceIdSearch,
} from '../../components/experiment-page/components/traces-v4/utils/buildTracesV4SearchParams';
import {
  compileFilterModel,
  compileTagFilters,
} from '../../components/experiment-page/components/traces-v4/utils/filterModel';
import { SEARCH_DEBOUNCE_MS } from '../../components/experiment-page/components/traces-v4/utils/constants';
import { useNumericAssessmentKeys } from '../../components/experiment-page/components/traces-v4/hooks/useNumericAssessmentKeys';
import { useEvalRunTracesV4UrlState } from './useEvalRunTracesV4UrlState';

const EXCLUDE_SCORER_TRACES_CLAUSE = 'tags.`mlflow.trace.sourceScorer` IS NULL';
const EVAL_RUN_DEFAULT_HIDDEN_COLUMNS: ReadonlySet<TraceColumnId> = new Set(['start_time']);

interface UseEvalRunTracesV4ControllerParams {
  experimentId: string;
  runUuid: string;
}

export interface UseEvalRunTracesV4ControllerResult {
  url: ReturnType<typeof useEvalRunTracesV4UrlState>;
  page: TracesPageQueryResult;
  columns: ReturnType<typeof useTracesV4Columns>;
  assessments: ReturnType<typeof useTracesV4AssessmentColumns>;
  columnSizing: ReturnType<typeof useTracesV4ColumnSizing>;
  bulk: ReturnType<typeof useBulkTraceSelection>;
  searchInput: ReturnType<typeof useDebouncedSearchInput>;
  filterModel: TraceFilterModel;
  setFilterModel: (next: TraceFilterModel) => void;
  activeFilterCount: number;
  /** Selected trace id, exposed only after the trace is known to belong to this run. */
  activeTraceId?: string;
  selectedTraceValidation: {
    isLoading: boolean;
    isFetching: boolean;
    error: unknown;
    refetch: () => void;
  };
  isGroupedBySession: boolean;
  goToPage: (target: number) => void;
  onFilterByTag: (key: string, value: string) => void;
  flags: {
    hasNoTracesAtAll: boolean;
    hasNoSearchResults: boolean;
    isEmptyPageBeyondFirst: boolean;
  };
}

/** The id the URL uses for a trace: the V4 long id for UC-backed traces, the bare id otherwise. */
const idOf = (trace: ModelTraceInfoV3): string =>
  doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;

export const buildEvalRunTraceFilter = ({
  runUuid,
  searchQuery,
  extraClauses,
}: {
  runUuid: string;
  searchQuery?: string;
  extraClauses?: string[];
}): string => {
  // The OSS `run_id` filter also matches traces linked to the run, not only the ones it created.
  const runClause = `attributes.run_id = '${escapeFilterValue(runUuid)}'`;
  const userFilter = buildFilter({ searchQuery, extraClauses });
  const requiredFilter = `${runClause} AND ${EXCLUDE_SCORER_TRACES_CLAUSE}`;
  return userFilter ? `${requiredFilter} AND ${userFilter}` : requiredFilter;
};

/** Owns the eval-run-specific trace query policy while reusing the shared table mechanisms. */
export const useEvalRunTracesV4Controller = ({
  experimentId,
  runUuid,
}: UseEvalRunTracesV4ControllerParams): UseEvalRunTracesV4ControllerResult => {
  const url = useEvalRunTracesV4UrlState();
  const queryClient = useQueryClient();
  const monitoringConfig = useMonitoringConfig();
  const [filterModel, setFilterModelState] = useState<TraceFilterModel>(EMPTY_FILTER_MODEL);
  const isGroupedBySession = shouldEnableSessionGrouping() && url.isGroupedBySession;
  const persistenceId = `${experimentId}.${runUuid}`;
  const { setPageIndex } = url;
  const setFilterModel = useCallback(
    (next: TraceFilterModel) => {
      setPageIndex(1);
      setFilterModelState(next);
    },
    [setPageIndex],
  );

  const searchInput = useDebouncedSearchInput({
    committedValue: url.search,
    onCommit: url.setSearch,
    debounceMs: SEARCH_DEBOUNCE_MS,
    commitOnChange: false,
  });

  const locations = useMemo<ModelTraceSearchLocation[]>(
    () => [createTraceLocationForExperiment(experimentId)],
    [experimentId],
  );
  const { numericAssessmentKeys, recordTraces } = useNumericAssessmentKeys(persistenceId);
  const filter = useMemo(
    () =>
      buildEvalRunTraceFilter({
        runUuid,
        searchQuery: url.search,
        extraClauses: [...compileFilterModel(filterModel, numericAssessmentKeys), ...compileTagFilters(url.tagFilters)],
      }),
    [runUuid, url.search, url.tagFilters, filterModel, numericAssessmentKeys],
  );
  const orderBy = useMemo(() => buildOrderBy(url.sort, url.dir), [url.sort, url.dir]);
  const queryPageSize = isGroupedBySession ? GROUPED_TRACES_LIMIT : url.pageSize;
  const identity = useMemo<TracesQueryIdentity>(
    () => ({ locations, filter, orderBy, pageSize: queryPageSize }),
    [locations, filter, orderBy, queryPageSize],
  );
  const tokenCache = useTraceTokenCache();
  const searchPage = useTracesPageQuery({
    identity,
    pageIndex: url.pageIndex,
    tokenCache,
    enabled: true,
    onPageIndexChange: url.setPageIndex,
  });
  // Show only the assessments this run produced; feedback other runs logged on a linked trace stays out.
  const runScopedTraces = useMemo(
    () => filterTracesByAssessmentSourceRunId(searchPage.traces, runUuid) ?? [],
    [searchPage.traces, runUuid],
  );
  useEffect(() => recordTraces(runScopedTraces), [runScopedTraces, recordTraces]);
  const page = useMemo<TracesPageQueryResult>(
    () => ({ ...searchPage, traces: runScopedTraces }),
    [searchPage, runScopedTraces],
  );

  const selectedTraceId = url.traceId;
  const isSelectedTraceIdValid = Boolean(selectedTraceId && isExactTraceIdSearch(selectedTraceId));
  const selectedTraceOnPage = useMemo(
    () =>
      selectedTraceId && !page.isPreviousData
        ? page.traces.find((trace) => selectedTraceId === trace.trace_id || selectedTraceId === idOf(trace))
        : undefined,
    [page.isPreviousData, page.traces, selectedTraceId],
  );
  const isSelectedTraceOnPage = Boolean(selectedTraceOnPage);
  const isPageSettled = !page.isLoading && !page.isFetching;
  const selectedTraceValidationEnabled = isPageSettled && isSelectedTraceIdValid && !isSelectedTraceOnPage;
  const selectedTraceValidationQuery = useQuery<ModelTraceInfoV3 | null, Error>({
    queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'eval-run-selected-trace-id', experimentId, runUuid, selectedTraceId],
    queryFn: async ({ signal }) => {
      if (!selectedTraceId) {
        return null;
      }
      const traces = await searchMlflowTracesQueryFn({
        signal,
        locations,
        filter: buildEvalRunTraceFilter({ runUuid, searchQuery: selectedTraceId }),
        pageSize: 1,
        limit: 1,
      });
      return filterTracesByAssessmentSourceRunId(traces, runUuid)?.[0] ?? null;
    },
    enabled: selectedTraceValidationEnabled,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const activeTrace = selectedTraceOnPage ?? selectedTraceValidationQuery.data ?? undefined;
  const activeTraceId = activeTrace ? idOf(activeTrace) : undefined;

  const { setTraceId } = url;
  useEffect(() => {
    if (!selectedTraceId || isSelectedTraceOnPage) {
      return;
    }
    const isValidatedMiss = selectedTraceValidationQuery.isSuccess && selectedTraceValidationQuery.data === null;
    if (!isSelectedTraceIdValid || isValidatedMiss) {
      setTraceId(undefined);
    }
  }, [
    isSelectedTraceIdValid,
    isSelectedTraceOnPage,
    selectedTraceId,
    selectedTraceValidationQuery.data,
    selectedTraceValidationQuery.isSuccess,
    setTraceId,
  ]);

  const lastRefreshTime = monitoringConfig.lastRefreshTime;
  const initialRefreshTimeRef = useRef(lastRefreshTime);
  useEffect(() => {
    if (lastRefreshTime === initialRefreshTimeRef.current) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
  }, [lastRefreshTime, queryClient]);

  const isCurrentPageKnown = tokenCache.isPageKnown(url.pageIndex);
  useEffect(() => {
    if (url.pageIndex > 1 && !page.isLoading && !isCurrentPageKnown) {
      setPageIndex(1);
    }
  }, [url.pageIndex, page.isLoading, isCurrentPageKnown, setPageIndex]);

  const hasSessionOnPage = useMemo(
    () => page.traces.some((trace) => Boolean(trace.trace_metadata?.[SESSION_ID_METADATA_KEY])),
    [page.traces],
  );
  const columns = useTracesV4Columns(persistenceId, {
    hasSessionOnPage,
    defaultHiddenColumns: EVAL_RUN_DEFAULT_HIDDEN_COLUMNS,
  });
  const assessments = useTracesV4AssessmentColumns(persistenceId, page.traces);
  const columnSizing = useTracesV4ColumnSizing(persistenceId);
  const bulk = useBulkTraceSelection(page.traces);

  const { clear: clearBulk } = bulk;
  useEffect(() => {
    clearBulk();
  }, [runUuid, url.search, url.sort, url.dir, url.tagFilters, filterModel, clearBulk]);

  const { submit: submitSearch } = searchInput;
  const { goToPage: pageGoToPage } = page;
  const goToPage = useCallback(
    (target: number) => {
      submitSearch();
      pageGoToPage(target);
    },
    [submitSearch, pageGoToPage],
  );

  const activeFilterCount = countActiveFilters(filterModel) + url.tagFilters.length;
  const hasActiveFilters = url.search.trim().length > 0 || activeFilterCount > 0;
  const hasNoResults = !page.isLoading && !page.isFetching && page.traces.length === 0;
  const isEmptyPageBeyondFirst = hasNoResults && url.pageIndex > 1;
  const hasNoTracesAtAll = hasNoResults && !hasActiveFilters && !isEmptyPageBeyondFirst;
  const hasNoSearchResults = hasNoResults && hasActiveFilters && !isEmptyPageBeyondFirst;

  return {
    url,
    page,
    columns,
    assessments,
    columnSizing,
    bulk,
    searchInput,
    filterModel,
    setFilterModel,
    activeFilterCount,
    activeTraceId,
    selectedTraceValidation: {
      isLoading: selectedTraceValidationEnabled && selectedTraceValidationQuery.isLoading,
      isFetching: selectedTraceValidationEnabled && selectedTraceValidationQuery.isFetching,
      error: selectedTraceValidationEnabled ? selectedTraceValidationQuery.error : null,
      refetch: selectedTraceValidationQuery.refetch,
    },
    isGroupedBySession,
    goToPage,
    onFilterByTag: url.addTagFilter,
    flags: { hasNoTracesAtAll, hasNoSearchResults, isEmptyPageBeyondFirst },
  };
};

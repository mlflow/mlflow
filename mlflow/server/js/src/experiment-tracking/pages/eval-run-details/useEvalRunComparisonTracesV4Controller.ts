import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { useIntl } from 'react-intl';
import {
  convertTraceInfoV3ToRunEvalEntry,
  createTraceLocationForExperiment,
  doesTraceSupportV4API,
  getAssessmentInfos,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
  searchMlflowTracesQueryFn,
  useGenAiExperimentRunsForComparison,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { filterTracesByAssessmentSourceRunId } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import {
  createTraceV4LongIdentifier,
  isFeedbackAssessment,
  isSessionLevelAssessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import {
  DEFAULT_PAGE_SIZE,
  type PageSize,
  type TraceColumnId,
  type TraceComparisonRow,
  type TraceFilterModel,
  type TracesComparisonRow,
} from '@databricks/web-shared/traces-table';
import { useDebouncedSearchInput } from '../../components/experiment-page/components/traces-v4/hooks/useDebouncedSearchInput';
import { useTracesV4AssessmentColumns } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4AssessmentColumns';
import { useTracesV4Columns } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Columns';
import { useTracesV4ColumnSizing } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4ColumnSizing';
import {
  assessmentNameFromColumnId,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { isExactTraceIdSearch } from '../../components/experiment-page/components/traces-v4/utils/buildTracesV4SearchParams';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { buildEvalRunTraceComparisonRows } from './evalRunTraceComparisonRows';
import { buildEvalRunTraceFilter } from './useEvalRunTracesV4Controller';
import { useEvalRunTracesV4UrlState } from './useEvalRunTracesV4UrlState';
import { useEvalRunComparisonRunTraces } from './useEvalRunComparisonRunTraces';
import { useEvalRunComparisonTraceInputs } from './useEvalRunComparisonTraceInputs';
import {
  getEvalRunComparisonColumnAggregates,
  getEvalRunComparisonOutcome,
  type EvalRunComparisonOutcome,
  type NumericAggregateMethod,
} from './evalRunComparisonColumnAggregates';
import { useEvalRunComparisonTracesUrlState } from './useEvalRunComparisonTracesUrlState';
import { useEvalRunComparisonFilterFields } from './useEvalRunComparisonFilterFields';
import { filterEvalRunComparisonRows, validateComparisonFilters } from './evalRunComparisonTraceFilters';
import { sortEvalRunComparisonRows } from './evalRunComparisonTraceSort';
import { buildEvalRunSessionComparisonRows, getComparisonRunTraces } from './evalRunSessionComparisonRows';
import {
  getEvalRunSessionComparisonColumnAggregates,
  getSessionComparisonAssessmentInfos,
  getSessionComparisonSummaries,
  sortSessionComparisonRows,
} from './evalRunComparisonSessionSummaries';

/** The id the URL uses for a trace, as on the single-run page: the long id when the trace supports it, else the bare id. */
export const getEvalRunComparisonTraceId = (trace: ModelTraceInfoV3): string =>
  doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;

const COMPARISON_DEFAULT_HIDDEN_COLUMNS: ReadonlySet<TraceColumnId> = new Set(['start_time']);

export type { NumericAggregateMethod } from './evalRunComparisonColumnAggregates';

export interface EvalRunComparisonOutcomeFilter {
  id: number;
  columnId: string;
  runUuid: string;
  outcome: EvalRunComparisonOutcome;
}

type OutcomeFilterSelection = Omit<EvalRunComparisonOutcomeFilter, 'id'>;

const isSameOutcomeFilterKey = (left: OutcomeFilterSelection, right: OutcomeFilterSelection) =>
  left.columnId === right.columnId && left.runUuid === right.runUuid;

const isSameOutcomeFilter = (left: OutcomeFilterSelection, right: OutcomeFilterSelection) =>
  isSameOutcomeFilterKey(left, right) && left.outcome === right.outcome;

export interface EvalRunComparisonTracesV4Params {
  experimentId: string;
  runUuids: string[];
  initialOutcomeFilter?: Omit<EvalRunComparisonOutcomeFilter, 'id'>;
  /** Outcome filter state owned by a parent that outlives this controller, e.g. across a mode remount. */
  outcomeFiltersState?: [EvalRunComparisonOutcomeFilter[], Dispatch<SetStateAction<EvalRunComparisonOutcomeFilter[]>>];
}

/** The keyed comparison composition owns local paging; query criteria use a separate URL namespace. */
export const useEvalRunComparisonTracesV4Controller = ({
  experimentId,
  runUuids,
  initialOutcomeFilter,
  outcomeFiltersState,
}: EvalRunComparisonTracesV4Params) => {
  const intl = useIntl();
  const url = useEvalRunTracesV4UrlState();
  const criteria = useEvalRunComparisonTracesUrlState();
  const persistenceId = `${experimentId}.run-comparison${criteria.isGroupedBySession ? '.sessions' : ''}`;
  const [collapsedComparisonGroups, setCollapsedComparisonGroups] = useState<ReadonlySet<string>>(() => new Set());
  const toggleComparisonGroup = useCallback((id: string) => {
    setCollapsedComparisonGroups((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);
  const [expandedComparisonRuns, setExpandedComparisonRuns] = useState<ReadonlyMap<string, ReadonlySet<string>>>(
    () => new Map(),
  );
  const toggleComparisonRun = useCallback((comparisonId: string, runUuid: string) => {
    setExpandedComparisonRuns((current) => {
      const next = new Map(current);
      const expandedRuns = new Set(current.get(comparisonId));
      if (expandedRuns.has(runUuid)) {
        expandedRuns.delete(runUuid);
      } else {
        expandedRuns.add(runUuid);
      }
      if (expandedRuns.size) {
        next.set(comparisonId, expandedRuns);
      } else {
        next.delete(comparisonId);
      }
      return next;
    });
  }, []);
  const [pageIndex, setPageIndex] = useState(1);
  const [pageSize, setPageSizeState] = useState<PageSize>(DEFAULT_PAGE_SIZE);
  const localOutcomeFiltersState = useState<EvalRunComparisonOutcomeFilter[]>(() =>
    initialOutcomeFilter ? [{ ...initialOutcomeFilter, id: 0 }] : [],
  );
  const [outcomeFilters, setOutcomeFilters] = outcomeFiltersState ?? localOutcomeFiltersState;
  const { isGroupedBySession, setIsGroupedBySession } = criteria;
  useEffect(() => {
    if (outcomeFilters.length && isGroupedBySession) setIsGroupedBySession(false);
  }, [outcomeFilters.length, isGroupedBySession, setIsGroupedBySession]);
  const [aggregateMethods, setAggregateMethods] = useState<Record<string, NumericAggregateMethod>>({});
  useEffect(() => {
    setPageIndex(1);
  }, [criteria.criteriaKey]);
  const setPageSize = useCallback((next: PageSize) => {
    setPageSizeState(next);
    setPageIndex(1);
  }, []);
  const searchInput = useDebouncedSearchInput({
    committedValue: criteria.search,
    onCommit: criteria.setSearch,
    debounceMs: 250,
    commitOnChange: false,
  });
  const data = useEvalRunComparisonRunTraces({ experimentId, runUuids });
  const { runInfos } = useGenAiExperimentRunsForComparison(experimentId);
  const runNames = useMemo(
    () =>
      new Map(runInfos?.flatMap((run) => (run.runUuid ? [[run.runUuid, run.runName ?? run.runUuid] as const] : []))),
    [runInfos],
  );
  const getRunColor = useGetExperimentRunColor();
  const allTraces = useMemo(() => data.runsWithTraces.flatMap((run) => run.traceInfos), [data.runsWithTraces]);
  const traceIds = useMemo(() => allTraces.map((trace) => trace.trace_id), [allTraces]);
  const inputs = useEvalRunComparisonTraceInputs({ experimentId, traceIds, enabled: data.isDataReady });
  const { refetch: refetchTraces } = data;
  const { refetch: refetchInputs } = inputs;
  const refetch = useCallback(async () => {
    await Promise.all([refetchTraces(), refetchInputs()]);
  }, [refetchTraces, refetchInputs]);
  // The toolbar's refresh button reloads the compared runs' traces, as on the single-run page.
  const queryClient = useQueryClient();
  const lastRefreshTime = useMonitoringConfig().lastRefreshTime;
  const initialRefreshTimeRef = useRef(lastRefreshTime);
  useEffect(() => {
    if (lastRefreshTime !== initialRefreshTimeRef.current) {
      void queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
    }
  }, [lastRefreshTime, queryClient]);
  const matchedRows = useMemo<TracesComparisonRow[]>(
    () =>
      inputs.isLoading
        ? []
        : isGroupedBySession
          ? buildEvalRunSessionComparisonRows(data.runsWithTraces, runNames, getRunColor, inputs.inputsByTraceId)
          : buildEvalRunTraceComparisonRows(data.runsWithTraces, runNames, getRunColor, '', inputs.inputsByTraceId),
    [data.runsWithTraces, runNames, getRunColor, inputs.isLoading, inputs.inputsByTraceId, isGroupedBySession],
  );
  // New sessions open automatically without overwriting a user's explicit collapse choices.
  const expandedComparisonGroups = useMemo(
    () =>
      new Set(
        matchedRows
          .filter((row) => row.kind === 'session' && !collapsedComparisonGroups.has(row.id))
          .map((row) => row.id),
      ),
    [matchedRows, collapsedComparisonGroups],
  );
  const filterFields = useEvalRunComparisonFilterFields(data.runsWithTraces, inputs.inputsByTraceId);
  const validateFilters = useCallback(
    (model: TraceFilterModel) => validateComparisonFilters(model, filterFields, intl),
    [filterFields, intl],
  );
  const filterError = criteria.invalidFilters
    ? intl.formatMessage({
        defaultMessage: 'The saved comparison filters could not be read. Clear them or apply new filters.',
        description: 'Malformed comparison filters in the URL',
      })
    : validateFilters(criteria.filterModel);
  const rows = useMemo(
    () =>
      filterError
        ? []
        : filterEvalRunComparisonRows(matchedRows, criteria.filterModel, criteria.search, inputs.inputsByTraceId),
    [matchedRows, criteria.filterModel, criteria.search, inputs.inputsByTraceId, filterError],
  );
  const traceRows = useMemo(() => rows.filter((row): row is TraceComparisonRow => row.kind !== 'session'), [rows]);
  const sessionAssessmentInfos = useMemo(
    () => (isGroupedBySession ? getSessionComparisonAssessmentInfos(allTraces, rows) : []),
    [allTraces, rows, isGroupedBySession],
  );
  const sessionSummaries = useMemo(
    () => getSessionComparisonSummaries(isGroupedBySession ? rows : [], sessionAssessmentInfos),
    [rows, sessionAssessmentInfos, isGroupedBySession],
  );
  const summaryRuns = useMemo(
    () =>
      runUuids.map((runUuid) => ({
        runUuid,
        label: runNames.get(runUuid) ?? runUuid,
        color: getRunColor(runUuid),
      })),
    [runUuids, runNames, getRunColor],
  );
  // Excluded rows and session feedback must not change the inferred summary type.
  const traceAssessmentInfos = useMemo(
    () =>
      getAssessmentInfos(
        intl,
        traceRows.flatMap((row) =>
          row.runs.flatMap(({ trace }) =>
            trace
              ? [
                  convertTraceInfoV3ToRunEvalEntry({
                    ...trace,
                    assessments: trace.assessments?.filter((assessment) => !isSessionLevelAssessment(assessment)),
                  }),
                ]
              : [],
          ),
        ),
        [],
      ),
    [traceRows, intl],
  );
  const filterOptionAggregates = useMemo(
    () =>
      isGroupedBySession
        ? getEvalRunSessionComparisonColumnAggregates(
            rows,
            sessionSummaries,
            sessionAssessmentInfos,
            intl,
            summaryRuns.length,
          )
        : getEvalRunComparisonColumnAggregates(
            traceRows,
            traceAssessmentInfos,
            intl,
            summaryRuns.length,
            aggregateMethods,
            true,
          ),
    [
      rows,
      traceRows,
      traceAssessmentInfos,
      intl,
      summaryRuns.length,
      aggregateMethods,
      isGroupedBySession,
      sessionSummaries,
      sessionAssessmentInfos,
    ],
  );
  useEffect(() => {
    if (isGroupedBySession || !data.isDataReady || inputs.isLoading) return;
    const filterableAssessments = new Set(
      traceAssessmentInfos
        .filter((info) => info.dtype === 'numeric' || info.dtype === 'boolean' || info.dtype === 'pass-fail')
        .map((info) => info.name),
    );
    setOutcomeFilters((current) => {
      const valid = current.filter(
        ({ columnId, runUuid }) =>
          runUuids.indexOf(runUuid) > 0 &&
          (!isAssessmentColumnId(columnId) || filterableAssessments.has(assessmentNameFromColumnId(columnId))),
      );
      return valid.length === current.length ? current : valid;
    });
  }, [isGroupedBySession, data.isDataReady, inputs.isLoading, runUuids, traceAssessmentInfos, setOutcomeFilters]);
  const toggleOutcomeFilter = useCallback(
    (filter: OutcomeFilterSelection) => {
      setOutcomeFilters((current) => {
        const existingIndex = current.findIndex((candidate) => isSameOutcomeFilterKey(candidate, filter));
        if (existingIndex < 0) {
          return [...current, { ...filter, id: Math.max(0, ...current.map(({ id }) => id)) + 1 }];
        }
        if (isSameOutcomeFilter(current[existingIndex], filter)) {
          return current.filter((candidate) => !isSameOutcomeFilterKey(candidate, filter));
        }
        return current.map((candidate, index) =>
          index === existingIndex ? { ...filter, id: candidate.id } : candidate,
        );
      });
      setPageIndex(1);
    },
    [setOutcomeFilters],
  );
  const removeOutcomeFilter = useCallback(
    (filter: EvalRunComparisonOutcomeFilter) => {
      setOutcomeFilters((current) => current.filter((candidate) => !isSameOutcomeFilter(candidate, filter)));
      setPageIndex(1);
    },
    [setOutcomeFilters],
  );
  const replaceOutcomeFilter = useCallback(
    (existing: EvalRunComparisonOutcomeFilter, next: EvalRunComparisonOutcomeFilter) => {
      setOutcomeFilters((current) =>
        current
          .filter((candidate) => candidate.id === existing.id || !isSameOutcomeFilterKey(candidate, next))
          .map((candidate) => (candidate.id === existing.id ? { ...next, id: existing.id } : candidate)),
      );
      setPageIndex(1);
    },
    [setOutcomeFilters],
  );
  const clearOutcomeFilters = useCallback(() => {
    setOutcomeFilters([]);
    setPageIndex(1);
  }, [setOutcomeFilters]);
  const setAggregateMethod = useCallback((columnId: string, method: NumericAggregateMethod) => {
    setAggregateMethods((current) => ({ ...current, [columnId]: method }));
  }, []);
  const filteredTraceRows = useMemo(
    () =>
      outcomeFilters.length
        ? traceRows.filter((row) =>
            outcomeFilters.every(
              (filter) =>
                getEvalRunComparisonOutcome(row, filter.columnId, filter.runUuid, traceAssessmentInfos) ===
                filter.outcome,
            ),
          )
        : traceRows,
    [traceRows, outcomeFilters, traceAssessmentInfos],
  );
  const columnAggregates = useMemo(
    () =>
      !isGroupedBySession && outcomeFilters.length
        ? getEvalRunComparisonColumnAggregates(
            filteredTraceRows,
            traceAssessmentInfos,
            intl,
            summaryRuns.length,
            aggregateMethods,
            true,
          )
        : filterOptionAggregates,
    [
      isGroupedBySession,
      outcomeFilters.length,
      filterOptionAggregates,
      filteredTraceRows,
      traceAssessmentInfos,
      intl,
      summaryRuns.length,
      aggregateMethods,
    ],
  );
  const sortedRows = useMemo<TracesComparisonRow[]>(
    () =>
      isGroupedBySession
        ? sortSessionComparisonRows(rows, sessionSummaries, criteria.sort, criteria.dir, sessionAssessmentInfos)
        : sortEvalRunComparisonRows(filteredTraceRows, criteria.sort, criteria.dir),
    [
      rows,
      filteredTraceRows,
      sessionSummaries,
      sessionAssessmentInfos,
      isGroupedBySession,
      criteria.sort,
      criteria.dir,
    ],
  );
  const totalRows = isGroupedBySession ? rows.length : filteredTraceRows.length;
  const lastPage = Math.max(1, Math.ceil(totalRows / pageSize));
  const currentPage = Math.min(pageIndex, lastPage);
  useEffect(() => {
    setPageIndex((current) => Math.min(current, lastPage));
  }, [lastPage]);
  const pageRows = useMemo(
    () => sortedRows.slice((currentPage - 1) * pageSize, currentPage * pageSize),
    [sortedRows, currentPage, pageSize],
  );
  const pageTraces = useMemo(
    () =>
      Array.from(
        new Map(
          pageRows.flatMap((row) =>
            row.runs.flatMap((run) =>
              getComparisonRunTraces(run).map((trace) => [getEvalRunComparisonTraceId(trace), trace] as const),
            ),
          ),
        ).values(),
      ),
    [pageRows],
  );
  const columns = useTracesV4Columns(persistenceId, {
    hasSessionOnPage: false,
    defaultHiddenColumns: COMPARISON_DEFAULT_HIDDEN_COLUMNS,
  });
  const assessmentTraces = useMemo(
    () =>
      isGroupedBySession
        ? allTraces.map((trace) => ({ ...trace, assessments: trace.assessments?.filter(isFeedbackAssessment) }))
        : allTraces,
    [allTraces, isGroupedBySession],
  );
  const assessments = useTracesV4AssessmentColumns(persistenceId, assessmentTraces, {
    includeSessionAssessments: isGroupedBySession,
  });
  const columnSizing = useTracesV4ColumnSizing(persistenceId);

  // A selected trace opens only when it belongs to one of the compared runs.
  const selectedTraceId = url.traceId;
  const isSelectedTraceIdValid = Boolean(selectedTraceId && isExactTraceIdSearch(selectedTraceId));
  const knownTrace = useMemo(
    () =>
      selectedTraceId && data.isDataReady
        ? allTraces.find(
            (trace) => trace.trace_id === selectedTraceId || getEvalRunComparisonTraceId(trace) === selectedTraceId,
          )
        : undefined,
    [allTraces, data.isDataReady, selectedTraceId],
  );
  const locations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const validateEnabled = Boolean(data.isDataReady && isSelectedTraceIdValid && !knownTrace);
  const validation = useQuery<ModelTraceInfoV3 | null, Error>({
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      'eval-run-comparison-selected-trace',
      experimentId,
      runUuids,
      selectedTraceId,
    ],
    queryFn: async ({ signal }) => {
      if (!selectedTraceId) {
        return null;
      }
      for (const runUuid of runUuids) {
        const traces = await searchMlflowTracesQueryFn({
          signal,
          locations,
          filter: buildEvalRunTraceFilter({ runUuid, searchQuery: selectedTraceId }),
          pageSize: 1,
          limit: 1,
        });
        const trace = filterTracesByAssessmentSourceRunId(traces, runUuid)?.[0];
        if (trace) {
          return trace;
        }
      }
      return null;
    },
    enabled: validateEnabled,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const activeTrace = knownTrace ?? (validateEnabled ? validation.data : undefined);
  const activeTraceId = activeTrace ? getEvalRunComparisonTraceId(activeTrace) : undefined;
  const { setTraceId } = url;
  useEffect(() => {
    if (
      selectedTraceId &&
      (!isSelectedTraceIdValid || (validateEnabled && validation.isSuccess && validation.data === null))
    ) {
      setTraceId(undefined);
    }
  }, [selectedTraceId, isSelectedTraceIdValid, validateEnabled, validation.isSuccess, validation.data, setTraceId]);

  return {
    data,
    refetch,
    url,
    persistenceId,
    searchInput,
    criteria,
    filterFields,
    validateFilters,
    filterError,
    columns,
    assessments,
    columnSizing,
    expandedComparisonGroups,
    toggleComparisonGroup,
    expandedComparisonRuns,
    toggleComparisonRun,
    sessionAssessmentInfos,
    rows: pageRows,
    assessmentInfos: traceAssessmentInfos,
    outcomeFilters,
    toggleOutcomeFilter,
    removeOutcomeFilter,
    replaceOutcomeFilter,
    clearOutcomeFilters,
    aggregateMethods,
    setAggregateMethod,
    filterOptionAggregates,
    columnAggregates,
    summaryRuns,
    allTraces,
    pageTraces,
    pageIndex: currentPage,
    pageSize,
    setPageIndex,
    setPageSize,
    hasNext: currentPage < lastPage,
    hasPrev: currentPage > 1,
    totalRows,
    hasFilters: Boolean(
      criteria.search || criteria.filterModel.length || criteria.invalidFilters || outcomeFilters.length,
    ),
    activeTraceId,
    validation,
    isCapped: data.fetchedTraceCounts.some((count) => count >= data.traceLimit),
    isLoading: data.isLoading || inputs.isLoading,
    isFetching: data.isFetching || inputs.isFetching,
  };
};

import { useCallback, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Button,
  CheckIcon,
  DropdownMenu,
  type InputRef,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  getTextColumnMaxSize,
  GROUPED_LEADING_COLUMNS,
  isTraceColumnId,
  type SessionHrefGetter,
  type TraceColumnId,
  type TraceComparisonRow,
  type TraceHrefGetter,
  type TracesTableViewProps,
  type TracesTableViewState,
} from '@databricks/web-shared/traces-table';
import { AssistantAwareDrawer } from '../../../common/components/AssistantAwareDrawer';
import { AssistantAwareActionBar } from '../../../common/components/AssistantAwareActionBar';
import { useLocation, useSearchParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { SELECTED_TRACE_ID_QUERY_PARAM } from '../../constants';
import { useSlashFocusSearch } from '../experiment-evaluation-datasets-v2/hooks/useSlashFocusSearch';
import { TracesV4TableSurface } from '../../components/experiment-page/components/traces-v4/components/TracesV4TableSurface';
import { useTracesV4ToolbarSlots } from '../../components/experiment-page/components/traces-v4/components/TracesV4Toolbar';
import { useTracesV4Density } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Density';
import { useTracesV4Notifications } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Notifications';
import { useTracesV4TraceActions } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4TraceActions';
import {
  assessmentColumnId,
  assessmentNameFromColumnId,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { EvalRunSelectedTraceErrorAlert, EvalRunTracesEmptyState } from './EvalRunTracesV4States';
import { buildEvalRunTableColumnOrder, EVAL_RUN_DEFAULT_COLUMN_SIZING } from './evalRunTracesV4TableConfig';
import {
  getEvalRunComparisonTraceId,
  useEvalRunComparisonTracesV4Controller,
  type EvalRunComparisonOutcomeFilter,
  type EvalRunComparisonTracesV4Params,
} from './useEvalRunComparisonTracesV4Controller';
import { EvalRunComparisonColumnSummary } from './EvalRunComparisonColumnSummary';
import { EvalRunComparisonOutcomeFilters } from './EvalRunComparisonOutcomeFilters';
import { withEvalRunComparisonAssessmentDeltas } from './evalRunComparisonAssessmentColumns';
import { buildComparisonSessionAssessmentColumns } from './EvalRunComparisonSessionAssessmentCell';

const NO_SELECTION = new Map<string, ModelTraceInfoV3>();
const NO_TRACES: ModelTraceInfoV3[] = [];
const noop = () => {};
const PREVIEW_LINES = { small: 1, standard: 2, tall: 6 } as const;
// The run name identifies each sub-row already, so the comparison never offers its column.
const isShownColumn = (id: TraceColumnId) => id !== 'run_name';

/** Moves `activeColumn` to `targetColumn`'s position in `order`. */
const moveColumn = (order: string[], activeColumn: string, targetColumn: string): string[] => {
  const next = [...order];
  const activeIndex = next.indexOf(activeColumn);
  const targetIndex = next.indexOf(targetColumn);
  if (activeIndex === -1 || targetIndex === -1) {
    return next;
  }
  next.splice(activeIndex, 1);
  next.splice(targetIndex, 0, activeColumn);
  return next;
};

export type EvalRunComparisonTracesV4TableProps = EvalRunComparisonTracesV4Params;

/** Matches the compared runs' traces by input (or sessions by scenario) and shows a sub-row per run. */
export const EvalRunComparisonTracesV4Table = (props: EvalRunComparisonTracesV4TableProps) => {
  const [searchParams] = useSearchParams();
  const sessionMode = searchParams.get('comparisonGroupBy') === 'session';
  // Outcome filters outlive the remount, so switching modes doesn't restore one the user removed.
  const outcomeFiltersState = useState<EvalRunComparisonOutcomeFilter[]>(() =>
    props.initialOutcomeFilter ? [{ ...props.initialOutcomeFilter, id: 0 }] : [],
  );
  // Storage hooks read their namespace at mount; do not carry trace preferences into session mode.
  return (
    <EvalRunComparisonTracesV4Content
      key={sessionMode ? 'sessions' : 'traces'}
      {...props}
      outcomeFiltersState={outcomeFiltersState}
    />
  );
};

const EvalRunComparisonTracesV4Content = (props: EvalRunComparisonTracesV4TableProps) => {
  const { experimentId } = props;
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const controller = useEvalRunComparisonTracesV4Controller(props);
  const { data, url, columns, assessments, columnSizing, searchInput, criteria } = controller;
  const {
    columnAggregates,
    filterOptionAggregates,
    summaryRuns,
    outcomeFilters,
    toggleOutcomeFilter,
    assessmentInfos,
    aggregateMethods,
    setAggregateMethod,
    allTraces,
  } = controller;
  const isGroupedBySession = criteria.isGroupedBySession;
  const { density, setDensity } = useTracesV4Density(controller.persistenceId);
  const { refetch } = controller;
  const refetchComparison = useCallback(() => void refetch(), [refetch]);
  const { notify, notificationContainer } = useTracesV4Notifications();
  const actions = useTracesV4TraceActions(experimentId, controller.pageTraces, refetchComparison, notify);
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);
  const location = useLocation();
  const getTraceHref = useCallback<TraceHrefGetter>(
    (trace) => {
      const params = new URLSearchParams(location.search);
      params.set('selectedEvaluationId', getEvalRunComparisonTraceId(trace));
      params.delete('traceId');
      params.delete('spanId');
      return `${location.pathname}?${params.toString()}${location.hash}`;
    },
    [location],
  );
  const onTraceSelected = useCallback(
    (trace: ModelTraceInfoV3) => url.setTraceId(getEvalRunComparisonTraceId(trace)),
    [url],
  );
  const getSessionHref = useCallback<SessionHrefGetter>(
    ({ trace, sessionId }) => {
      const base = Routes.getExperimentPageTabSingleChatSessionRoute(experimentId, sessionId);
      return `${base}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: trace.trace_id }).toString()}`;
    },
    [experimentId],
  );
  const visibleColumns = useMemo<TraceColumnId[]>(() => {
    const visible = columns.visibleColumns.filter(isShownColumn);
    return isGroupedBySession ? [...new Set<TraceColumnId>(['session', ...visible, 'input', 'output'])] : visible;
  }, [columns.visibleColumns, isGroupedBySession]);
  const filterableColumns = useMemo(
    () => [
      { id: 'state', label: intl.formatMessage({ defaultMessage: 'State', description: 'Trace state column' }) },
      {
        id: 'duration',
        label: intl.formatMessage({ defaultMessage: 'Duration', description: 'Trace duration column' }),
      },
      { id: 'tokens', label: intl.formatMessage({ defaultMessage: 'Tokens', description: 'Trace tokens column' }) },
      { id: 'cost', label: intl.formatMessage({ defaultMessage: 'Cost', description: 'Trace cost column' }) },
      ...assessmentInfos
        .filter((info) => info.dtype === 'numeric' || info.dtype === 'boolean' || info.dtype === 'pass-fail')
        .map((info) => ({ id: assessmentColumnId(info.name), label: info.name })),
    ],
    [intl, assessmentInfos],
  );
  const columnLabels = useMemo(
    () => new Map(filterableColumns.map(({ id, label }) => [id, label])),
    [filterableColumns],
  );
  const traceAssessmentColumnDefs = useMemo(
    () =>
      assessments.columnDefs.map((column) => {
        const columnId = column.id;
        if (!columnId) return column;
        const info = assessmentInfos.find((item) => item.name === assessmentNameFromColumnId(columnId));
        if (info?.dtype !== 'numeric') return column;
        const current = aggregateMethods[columnId] ?? 'mean';
        return {
          ...column,
          headerMenuContent: (
            <>
              {(['mean', 'median', 'min', 'max'] as const).map((method) => (
                <DropdownMenu.Item
                  key={method}
                  componentId="mlflow.eval-run-comparison.aggregate-method"
                  onClick={() => setAggregateMethod(columnId, method)}
                >
                  {method === 'mean' ? (
                    <FormattedMessage defaultMessage="Mean" description="Mean comparison aggregate option" />
                  ) : method === 'median' ? (
                    <FormattedMessage defaultMessage="Median" description="Median comparison aggregate option" />
                  ) : method === 'min' ? (
                    <FormattedMessage defaultMessage="Min" description="Minimum comparison aggregate option" />
                  ) : (
                    <FormattedMessage defaultMessage="Max" description="Maximum comparison aggregate option" />
                  )}
                  {current === method && (
                    <DropdownMenu.HintColumn>
                      <CheckIcon />
                    </DropdownMenu.HintColumn>
                  )}
                </DropdownMenu.Item>
              ))}
            </>
          ),
        };
      }),
    [assessments.columnDefs, assessmentInfos, aggregateMethods, setAggregateMethod],
  );
  const extraColumns = useMemo(
    () =>
      isGroupedBySession
        ? buildComparisonSessionAssessmentColumns(assessments.columnDefs, controller.sessionAssessmentInfos)
        : withEvalRunComparisonAssessmentDeltas(
            traceAssessmentColumnDefs,
            allTraces,
            controller.rows.filter((row): row is TraceComparisonRow => row.kind !== 'session'),
            intl,
          ),
    [
      assessments.columnDefs,
      traceAssessmentColumnDefs,
      controller.sessionAssessmentInfos,
      isGroupedBySession,
      allTraces,
      controller.rows,
      intl,
    ],
  );
  const columnHeaderSummaries = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(columnAggregates).map(([columnId, aggregate]) => [
          columnId,
          {
            content: (
              <EvalRunComparisonColumnSummary
                aggregate={aggregate}
                runs={summaryRuns}
                isGroupedBySession={isGroupedBySession}
                columnLabel={columnLabels.get(columnId) ?? assessmentNameFromColumnId(columnId)}
                higherIsBetter={isAssessmentColumnId(columnId)}
                filterOptionCounts={filterOptionAggregates[columnId]?.values.map(({ changeCounts }) => changeCounts)}
                onOutcomeClick={
                  isGroupedBySession
                    ? undefined
                    : (runUuid, outcome) => toggleOutcomeFilter({ columnId, runUuid, outcome })
                }
                activeOutcomes={Object.fromEntries(
                  outcomeFilters
                    .filter((filter) => filter.columnId === columnId)
                    .map((filter) => [filter.runUuid, filter.outcome]),
                )}
              />
            ),
            minWidth: Math.max(
              100,
              ...aggregate.values.map((value, index) => {
                const counts =
                  index > 0 && value.changeCounts
                    ? [value.changeCounts.improved, value.changeCounts.regressed].filter((count) => count > 0)
                    : [];
                const displayedText = [
                  value.displayValue,
                  index === 0 ? aggregate.label : value.displayDelta,
                  ...counts.map((count) => intl.formatNumber(count)),
                ]
                  .filter(Boolean)
                  .join(' ');
                return getTextColumnMaxSize(
                  [displayedText],
                  100,
                  theme.spacing.mid + theme.spacing.sm * 2 + counts.length * theme.typography.fontSizeSm,
                );
              }),
            ),
          },
        ]),
      ),
    [
      columnAggregates,
      filterOptionAggregates,
      summaryRuns,
      outcomeFilters,
      toggleOutcomeFilter,
      columnLabels,
      intl,
      theme,
      isGroupedBySession,
    ],
  );
  const comparisonColumnSizing = useMemo(
    () => ({ ...EVAL_RUN_DEFAULT_COLUMN_SIZING, ...columnSizing.columnSizing }),
    [columnSizing.columnSizing],
  );
  const defaultColumnOrder = useMemo(
    () =>
      buildEvalRunTableColumnOrder(
        columns.columnOrder,
        assessments.visibleIds,
        allTraces,
        controller.sessionAssessmentInfos.map((info) => info.name),
      ).filter((id) => !isTraceColumnId(id) || isShownColumn(id)),
    [assessments.visibleIds, columns.columnOrder, allTraces, controller.sessionAssessmentInfos],
  );
  // A drag in the column selector reorders this comparison until it changes, as on the single-run page.
  const [reorderedColumns, setReorderedColumns] = useState<string[]>();
  const comparisonColumnOrder = useMemo(
    () =>
      reorderedColumns
        ? [...new Set([...reorderedColumns.filter((id) => defaultColumnOrder.includes(id)), ...defaultColumnOrder])]
        : defaultColumnOrder,
    [defaultColumnOrder, reorderedColumns],
  );
  const reorderColumn = useCallback(
    (activeColumn: string, targetColumn: string) =>
      setReorderedColumns(moveColumn(comparisonColumnOrder, activeColumn, targetColumn)),
    [comparisonColumnOrder],
  );
  const resetColumns = useCallback(() => {
    columns.resetToDefaults();
    assessments.reset();
    setReorderedColumns(undefined);
  }, [columns, assessments]);
  const hideColumn = useCallback(
    (id: string) => {
      if (isAssessmentColumnId(id)) {
        assessments.toggle(id);
      } else if (isTraceColumnId(id)) {
        columns.toggleColumn(id);
      }
    },
    [columns, assessments],
  );
  const { validation } = controller;
  const refresh = useCallback(() => {
    void refetch();
    if (validation.isError) {
      void validation.refetch();
    }
  }, [refetch, validation]);
  const activeFilterCount = criteria.filterModel.length || (criteria.invalidFilters ? 1 : 0);
  const toolbar = useTracesV4ToolbarSlots({
    filterModel: criteria.filterModel,
    onFilterChange: criteria.setFilterModel,
    onClearFilters: criteria.clearFilters,
    activeFilterCount,
    filterFields: controller.filterFields,
    validateFilters: controller.validateFilters,
    filterDescription: isGroupedBySession
      ? intl.formatMessage({
          defaultMessage:
            'Keep every turn and run in a session comparison when one trace matches all filters and the search. Filters use individual trace values, not session scores or totals.',
          description: 'Trace-level filtering semantics in the session comparison',
        })
      : intl.formatMessage({
          defaultMessage: 'Keep a comparison row when one trace matches all filters and the search.',
          description: 'Comparison filter semantics shown in the filter builder',
        }),
    // Outcome filters only match traces, so session grouping is offered only without them.
    showSessionGrouping: outcomeFilters.length === 0,
    groupedLockedColumns: GROUPED_LEADING_COLUMNS,
    visibleColumns,
    columnOrder: comparisonColumnOrder,
    onReorderColumn: reorderColumn,
    onToggleColumn: columns.toggleColumn,
    onResetColumns: resetColumns,
    assessmentColumns: assessments,
    sort: isTraceColumnId(criteria.sort) ? criteria.sort : 'start_time',
    dir: criteria.dir,
    onSort: criteria.setSort,
    density,
    onDensityChange: setDensity,
    selectionCount: 0,
    onBulkDelete: noop,
    isRefreshing: controller.isFetching,
    experimentId,
    actions,
    selectedTraceInfos: NO_TRACES,
    isGroupedBySession,
    onToggleSessionGrouping: criteria.setIsGroupedBySession,
    showSessionViewSwitch: true,
  });
  const error = useMemo(
    () =>
      data.failedRunUuids.length
        ? new Error(
            intl.formatMessage({
              defaultMessage: 'Could not load traces for all selected runs. Retry to load the comparison.',
              description: 'Comparison cannot be shown when a run trace query fails',
            }),
          )
        : undefined,
    [data.failedRunUuids, intl],
  );
  const viewState: TracesTableViewState = error
    ? 'error'
    : controller.isLoading || !data.isDataReady
      ? 'loading'
      : controller.totalRows
        ? 'ready'
        : controller.hasFilters
          ? 'no-results'
          : 'empty';
  const tableViewProps: TracesTableViewProps = {
    viewState,
    traces: controller.pageTraces,
    comparisonRows: controller.rows,
    expandedComparisonGroups: controller.expandedComparisonGroups,
    onToggleComparisonGroup: controller.toggleComparisonGroup,
    expandedComparisonRuns: controller.expandedComparisonRuns,
    onToggleComparisonRun: controller.toggleComparisonRun,
    isGroupedBySession,
    columnHeaderSummaries,
    visibleColumns,
    extraColumns,
    columnOrder: comparisonColumnOrder,
    initialColumnSizing: comparisonColumnSizing,
    onColumnSizingSettled: columnSizing.setColumnSizing,
    isLoading: controller.isLoading || !data.isDataReady,
    isFetching: controller.isFetching,
    skeletonRowCount: controller.pageSize,
    onTraceSelected,
    getTraceHref,
    getSessionHref,
    selectedTraceId: controller.activeTraceId,
    selectedForBulk: NO_SELECTION,
    isAllOnPageSelected: false,
    isSomeOnPageSelected: false,
    onToggleBulkRow: noop,
    onToggleBulkAll: noop,
    sort: criteria.sort,
    dir: criteria.dir,
    onSort: criteria.setSort,
    enableSorting: true,
    sortTooltip: isGroupedBySession
      ? intl.formatMessage({
          defaultMessage:
            'Groups are sorted by the baseline session’s summary. Ties use the other runs in displayed order. Missing values appear last. Turns remain chronological.',
          description: 'Explains sorting complete matched sessions',
        })
      : intl.formatMessage({
          defaultMessage:
            'Groups are sorted by the baseline trace’s value. Ties use the other runs in displayed order. Missing values appear last.',
          description: 'Explains baseline-first comparison group sorting on the active column sort toggle',
        }),
    onHideColumn: hideColumn,
    previewLineClamp: PREVIEW_LINES[density],
    searchValue: searchInput.input,
    searchPlaceholder: intl.formatMessage({
      defaultMessage: 'Search traces by id or input',
      description: 'Search input placeholder for matched traces in an evaluation-run comparison',
    }),
    onSearchChange: searchInput.setInput,
    onSearchClear: searchInput.clear,
    onSearchSubmit: searchInput.submit,
    searchInputRef,
    leftControls: toolbar.sessionViewSwitch,
    rightControls: toolbar.rightControls,
    pageIndex: controller.pageIndex,
    pageSize: controller.pageSize,
    onPageChange: controller.setPageIndex,
    onPageSizeChange: controller.setPageSize,
    hasNext: controller.hasNext,
    hasPrev: controller.hasPrev,
    paginationLeadingContent: (
      <Typography.Text color="secondary">
        {isGroupedBySession ? (
          <FormattedMessage
            defaultMessage="{count, plural, one {# comparison group} other {# comparison groups}}"
            description="Number of matched session or standalone trace groups"
            values={{ count: controller.totalRows }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="{count, plural, one {# comparison row} other {# comparison rows}}"
            description="Number of matched input rows in the run comparison"
            values={{ count: controller.totalRows }}
          />
        )}
      </Typography.Text>
    ),
    PaginationBarWrapper: AssistantAwareActionBar,
    onClearFilters: () => {
      criteria.clearAll();
      controller.clearOutcomeFilters();
    },
    onRetry: refresh,
    error,
    customEmptyState: viewState === 'empty' ? <EvalRunTracesEmptyState /> : undefined,
    bannerSlot: (
      <>
        {!isGroupedBySession && (
          <EvalRunComparisonOutcomeFilters
            filters={outcomeFilters}
            aggregates={filterOptionAggregates}
            columns={filterableColumns}
            runs={summaryRuns}
            onReplace={controller.replaceOutcomeFilter}
            onRemove={controller.removeOutcomeFilter}
            onClear={controller.clearOutcomeFilters}
          />
        )}
        {controller.filterError && (
          <Alert
            componentId="mlflow.eval-run-comparison.invalid-filters"
            type="error"
            message={controller.filterError}
            action={
              <Button componentId="mlflow.eval-run-comparison.clear-invalid-filters" onClick={criteria.clearFilters}>
                <FormattedMessage defaultMessage="Clear filters" description="Clear invalid comparison filters" />
              </Button>
            }
          />
        )}
        {controller.isCapped && (
          <Alert
            componentId="mlflow.eval-run-comparison.traces-cap"
            type="warning"
            message={
              isGroupedBySession ? (
                <FormattedMessage
                  defaultMessage="Comparison is limited to {limit, number} traces per run. Sessions may have missing turns, and matches and summaries may be incomplete."
                  description="Loaded session summaries may be partial when the trace cap is reached"
                  values={{ limit: data.traceLimit }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Comparison is limited to {limit, number} traces per run. Matches and results may be incomplete."
                  description="Warning when the run comparison trace fetch limit is reached"
                  values={{ limit: data.traceLimit }}
                />
              )
            }
          />
        )}
        {validation.isError && <EvalRunSelectedTraceErrorAlert onRetry={validation.refetch} />}
      </>
    ),
  };
  return (
    <TracesV4TableSurface
      tableViewProps={tableViewProps}
      providerProps={{
        modelTraceExplorer: {
          renderExportTracesToDatasetsModal: actions.renderExportTracesToDatasetsModal,
          renderAddToReviewQueueDropdown: actions.AddToReviewQueueDropdown,
          DrawerComponent: AssistantAwareDrawer,
        },
        genAITracesTable: { experimentId, getTrace: actions.getTrace, isGroupedBySession },
      }}
      drawerProps={{
        traceId: controller.activeTraceId,
        onClose: () => url.setTraceId(undefined),
        experimentId,
        traces: controller.pageTraces,
        onSelectTrace: url.setTraceId,
        runJudgeConfiguration: actions.runJudges?.runJudgeConfiguration,
      }}
      overlaySlots={{ actionModals: actions.runJudges?.RunJudgesModal, notificationContainer }}
      layoutProps={{ paddingTop: 0, paddingLeft: 0, marginBottom: -theme.spacing.sm }}
    />
  );
};

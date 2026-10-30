import { useCallback, useMemo, useRef, useState } from 'react';
import { type InputRef, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { createTraceV4LongIdentifier, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  doesTraceSupportV4API,
  type AssessmentFilter,
  type AssessmentValueType,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { ERROR_KEY } from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import {
  EMPTY_FILTER_MODEL,
  FilterOp,
  TRACE_COLUMN_IDS,
  TracesErrorAlert,
  type SessionHrefGetter,
  type SessionSelectionHandler,
  type TraceColumnId,
  type TraceHrefGetter,
  type TracesTableViewProps,
  type TracesTableViewState,
} from '@databricks/web-shared/traces-table';
import { AssistantAwareDrawer } from '../../../common/components/AssistantAwareDrawer';
import { AssistantAwareActionBar } from '../../../common/components/AssistantAwareActionBar';
import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { useRegisterSelectedIds } from '../../../assistant';
import Routes from '../../routes';
import { SELECTED_TRACE_ID_QUERY_PARAM } from '../../constants';
import { useSlashFocusSearch } from '../experiment-evaluation-datasets-v2/hooks/useSlashFocusSearch';
import { useDeleteTracesMutation } from '../../components/evaluations/hooks/useDeleteTraces';
import { useTracesV4Density } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Density';
import { useTracesV4Notifications } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4Notifications';
import { useTracesV4TraceActions } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4TraceActions';
import { useTracesV4UnifiedSearch } from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4UnifiedSearch';
import {
  getSessionAssessmentNames,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { useTracesV4ToolbarSlots } from '../../components/experiment-page/components/traces-v4/components/TracesV4Toolbar';
import { TracesV4DeleteModal } from '../../components/experiment-page/components/traces-v4/components/TracesV4DeleteModal';
import { makeTracesV4ErrorDescription } from '../../components/experiment-page/components/traces-v4/components/TracesV4States';
import {
  TracesV4TableSurface,
  type TracesV4TableSurfaceProviderProps,
} from '../../components/experiment-page/components/traces-v4/components/TracesV4TableSurface';
import type { TracesV4TraceDrawerProps } from '../../components/experiment-page/components/traces-v4/components/TracesV4TraceDrawer';
import { useEvalRunTracesV4Controller } from './useEvalRunTracesV4Controller';
import { EvalRunSelectedTraceErrorAlert, EvalRunTracesEmptyState } from './EvalRunTracesV4States';
import { buildEvalRunTableColumnOrder, EVAL_RUN_DEFAULT_COLUMN_SIZING } from './evalRunTracesV4TableConfig';
import { useEvalRunAssessmentAggregates } from './useEvalRunAssessmentAggregates';
import {
  buildEvalRunAssessmentColumnDefs,
  EvalRunAssessmentHeaderToggle,
  hasCollapsibleAssessmentChart,
} from './EvalRunAssessmentColumnHeader';

export interface EvalRunTracesV4TableProps {
  experimentId: string;
  runUuid: string;
  /** The experiment's baseline run, when it's another run; the scorer headers show the change from it. */
  baselineRunUuid?: string;
  /** Saved run metrics of this run and of the baseline run, from `getEvalRunScores`. */
  currentScores?: Record<string, number>;
  baselineScores?: Record<string, number>;
}

const STANDARD_COLUMN_IDS: readonly string[] = TRACE_COLUMN_IDS;
const isStandardColumnId = (id: string): id is TraceColumnId => STANDARD_COLUMN_IDS.includes(id);
const PREVIEW_LINE_CLAMP_BY_DENSITY = {
  small: 1,
  standard: 2,
  tall: 6,
} as const;

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

/** Evaluation-run composition over the shared V4 traces rendering surface. */
export const EvalRunTracesV4Table = ({
  experimentId,
  runUuid,
  baselineRunUuid,
  currentScores,
  baselineScores,
}: EvalRunTracesV4TableProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const traceLinkLocation = useLocation();
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);
  const { notify, notificationContainer } = useTracesV4Notifications();

  const persistenceId = `${experimentId}.${runUuid}`;
  const assessmentSummaries = useEvalRunAssessmentAggregates({ experimentId, runUuid });
  const seededAssessmentNames = useMemo(
    () => assessmentSummaries.scorers.map((scorer) => scorer.name),
    [assessmentSummaries.scorers],
  );
  const controller = useEvalRunTracesV4Controller({ experimentId, runUuid, seededAssessmentNames });
  const { url, page, columns, assessments, columnSizing, bulk, searchInput, filterModel, flags } = controller;
  const { density, setDensity } = useTracesV4Density(persistenceId);
  // Edits, judge runs and deletes change the scores, so they reload the header summaries with the rows.
  const { refetch: refetchPage } = page;
  const { refetch: refetchAssessmentSummaries } = assessmentSummaries;
  const refetchTracesAndSummaries = useCallback(() => {
    refetchPage();
    refetchAssessmentSummaries();
  }, [refetchPage, refetchAssessmentSummaries]);
  const actions = useTracesV4TraceActions(experimentId, page.traces, refetchTracesAndSummaries, notify);

  // Header summaries: the scorer's rate or value over the run, with a chart for pass/fail scores. A chart
  // click toggles a filter on that value.
  const [areAssessmentChartsCollapsed, setAreAssessmentChartsCollapsed] = useState(false);
  const toggleAssessmentCharts = useCallback(() => setAreAssessmentChartsCollapsed((collapsed) => !collapsed), []);
  const scorersByName = useMemo(
    () => new Map(assessmentSummaries.scorers.map((scorer) => [scorer.name, scorer])),
    [assessmentSummaries.scorers],
  );
  const assessmentFilters = useMemo<AssessmentFilter[]>(
    () =>
      filterModel.flatMap((clause) => {
        if (clause.field !== 'assessment' || !clause.key || clause.operator !== FilterOp.EQUALS) {
          return [];
        }
        const dtype = scorersByName.get(clause.key)?.dtype;
        const filterValue =
          dtype === 'boolean' ? clause.value === 'true' : dtype === 'numeric' ? Number(clause.value) : clause.value;
        return [{ assessmentName: clause.key, filterValue, run: runUuid }];
      }),
    [filterModel, scorersByName, runUuid],
  );
  const { setFilterModel } = controller;
  const toggleAssessmentFilter = useCallback(
    (assessmentName: string, filterValue: AssessmentValueType) => {
      // The trace search can't match a missing or errored assessment, so those rows don't filter.
      if (filterValue === undefined || filterValue === ERROR_KEY) {
        return;
      }
      const value = String(filterValue);
      const otherClauses = filterModel.filter(
        (clause) => clause.field !== 'assessment' || clause.key !== assessmentName,
      );
      const isActive = filterModel.some(
        (clause) =>
          clause.field === 'assessment' &&
          clause.key === assessmentName &&
          clause.operator === FilterOp.EQUALS &&
          clause.value === value,
      );
      setFilterModel(
        isActive
          ? otherClauses
          : [...otherClauses, { field: 'assessment', key: assessmentName, operator: FilterOp.EQUALS, value }],
      );
    },
    [filterModel, setFilterModel],
  );
  const assessmentColumnDefs = useMemo(
    () =>
      buildEvalRunAssessmentColumnDefs({
        columnDefs: assessments.columnDefs,
        scorersByName,
        assessmentFilters,
        toggleAssessmentFilter,
        runUuid,
        areChartsCollapsed: areAssessmentChartsCollapsed,
        currentScores,
        baselineScores,
        baselineRunHref: baselineRunUuid
          ? Routes.getEvaluationRunDetailsRoute(experimentId, baselineRunUuid)
          : undefined,
      }),
    [
      assessments.columnDefs,
      scorersByName,
      assessmentFilters,
      toggleAssessmentFilter,
      runUuid,
      areAssessmentChartsCollapsed,
      currentScores,
      baselineScores,
      baselineRunUuid,
      experimentId,
    ],
  );
  const assessmentHeaderBottomOverlay = useMemo(
    () =>
      hasCollapsibleAssessmentChart(assessmentColumnDefs) ? (
        <EvalRunAssessmentHeaderToggle
          areChartsCollapsed={areAssessmentChartsCollapsed}
          onToggle={toggleAssessmentCharts}
        />
      ) : undefined,
    [assessmentColumnDefs, areAssessmentChartsCollapsed, toggleAssessmentCharts],
  );

  // Assessments lead the default order so scores are on the first screen. A drag in the column selector
  // reorders this view until the run changes.
  const defaultColumnOrder = useMemo(
    () =>
      buildEvalRunTableColumnOrder(
        columns.columnOrder,
        assessments.visibleIds,
        page.traces,
        // Session-level judges are feedback too, so they sit with the other feedback columns.
        getSessionAssessmentNames(page.traces),
      ),
    [assessments.visibleIds, columns.columnOrder, page.traces],
  );
  const [reorderedColumns, setReorderedColumns] = useState<string[]>();
  const evalRunColumnOrder = useMemo(
    () =>
      reorderedColumns
        ? [...new Set([...reorderedColumns.filter((id) => defaultColumnOrder.includes(id)), ...defaultColumnOrder])]
        : defaultColumnOrder,
    [defaultColumnOrder, reorderedColumns],
  );
  const reorderColumn = useCallback(
    (activeColumn: string, targetColumn: string) =>
      setReorderedColumns(moveColumn(evalRunColumnOrder, activeColumn, targetColumn)),
    [evalRunColumnOrder],
  );
  const evalRunColumnSizing = useMemo(
    () => ({ ...EVAL_RUN_DEFAULT_COLUMN_SIZING, ...columnSizing.columnSizing }),
    [columnSizing.columnSizing],
  );
  // The run name is the page's own breadcrumb, so its column is never offered here.
  const visibleColumns = useMemo(
    () => columns.visibleColumns.filter((column) => column !== 'run_name'),
    [columns.visibleColumns],
  );
  const tableColumnOrder = useMemo(
    () => evalRunColumnOrder.filter((column) => column !== 'run_name'),
    [evalRunColumnOrder],
  );

  const selectedTraceInfos = useMemo(() => Array.from(bulk.selected.values()), [bulk.selected]);
  const selectedTraceIds = useMemo(
    () => Object.fromEntries(Array.from(bulk.selected.keys()).map((id) => [id, true])),
    [bulk.selected],
  );
  useRegisterSelectedIds('selectedTraceIds', selectedTraceIds);

  const resetColumns = useCallback(() => {
    columns.resetToDefaults();
    assessments.reset();
    setReorderedColumns(undefined);
  }, [columns, assessments]);
  const handleHideColumn = useCallback(
    (columnId: string) => {
      if (isAssessmentColumnId(columnId)) {
        assessments.toggle(columnId);
      } else if (isStandardColumnId(columnId)) {
        columns.toggleColumn(columnId);
      }
    },
    [assessments, columns],
  );

  const handleTraceSelected = useCallback(
    (trace: ModelTraceInfoV3) =>
      url.setTraceId(doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id),
    [url],
  );
  const closeDrawer = useCallback(() => url.setTraceId(undefined), [url]);
  const getTraceHref = useCallback<TraceHrefGetter>(
    (trace) => {
      const traceId = doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;
      const searchParams = new URLSearchParams(traceLinkLocation.search);
      searchParams.set('selectedEvaluationId', traceId);
      searchParams.delete('traceId');
      searchParams.delete('spanId');
      return `${traceLinkLocation.pathname}?${searchParams.toString()}${traceLinkLocation.hash}`;
    },
    [traceLinkLocation],
  );
  const getSessionHref = useCallback<SessionHrefGetter>(
    ({ trace, sessionId }) => {
      const baseUrl = Routes.getExperimentPageTabSingleChatSessionRoute(experimentId, sessionId);
      return trace.trace_id
        ? `${baseUrl}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: trace.trace_id }).toString()}`
        : baseUrl;
    },
    [experimentId],
  );
  const handleSessionSelected = useCallback<SessionSelectionHandler>(
    (session) => {
      const destination = getSessionHref(session);
      if (destination) {
        navigate(destination);
      }
    },
    [getSessionHref, navigate],
  );

  const clearAllFilters = useCallback(() => {
    controller.setFilterModel(EMPTY_FILTER_MODEL);
    url.clearTagFilters();
  }, [controller, url]);
  const clearFilters = useCallback(() => {
    searchInput.clear();
    clearAllFilters();
  }, [searchInput, clearAllFilters]);

  const deleteTracesMutation = useDeleteTracesMutation();
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Snapshot ids when the modal opens so a refetch or a cleared selection can't change what is deleted.
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);
  const openDelete = useCallback(() => {
    setPendingDeleteIds(Array.from(bulk.selected.keys()));
    setDeleteOpen(true);
  }, [bulk.selected]);
  const cancelDelete = useCallback(() => {
    setDeleteOpen(false);
    setPendingDeleteIds([]);
  }, []);
  const confirmDelete = useCallback(() => {
    if (pendingDeleteIds.length === 0) {
      return;
    }
    const ids = pendingDeleteIds;
    deleteTracesMutation.mutate(
      { experimentId, traceRequestIds: ids },
      {
        onSuccess: () => {
          bulk.clear();
          setDeleteOpen(false);
          setPendingDeleteIds([]);
          refetchTracesAndSummaries();
          notify.success(
            intl.formatMessage(
              {
                defaultMessage: 'Deleted {count, plural, one {# trace} other {# traces}}',
                description: 'Success toast after bulk-deleting traces from an evaluation run',
              },
              { count: ids.length },
            ),
          );
        },
        onError: (err) => notify.error(err),
      },
    );
  }, [pendingDeleteIds, deleteTracesMutation, experimentId, bulk, refetchTracesAndSummaries, notify, intl]);

  const unifiedSearch = useTracesV4UnifiedSearch({
    filterModel,
    setFilterModel: controller.setFilterModel,
    tagFilters: url.tagFilters,
    setTagFilters: url.setTagFilters,
    searchInput,
  });

  const toolbarSlots = useTracesV4ToolbarSlots({
    filterModel: unifiedSearch.searchFilterModel,
    onFilterChange: unifiedSearch.setSearchFilters,
    hideFilterButton: true,
    traces: page.traces,
    onClearFilters: clearAllFilters,
    activeFilterCount: controller.activeFilterCount,
    visibleColumns,
    columnOrder: tableColumnOrder,
    onReorderColumn: reorderColumn,
    onToggleColumn: columns.toggleColumn,
    onResetColumns: resetColumns,
    assessmentColumns: assessments,
    sort: url.sort,
    dir: url.dir,
    onSort: url.setSort,
    density,
    onDensityChange: setDensity,
    selectionCount: bulk.selected.size,
    onBulkDelete: openDelete,
    isRefreshing: page.isFetching && !page.isLoading,
    experimentId,
    actions,
    selectedTraceInfos,
    isGroupedBySession: controller.isGroupedBySession,
    onToggleSessionGrouping: url.setIsGroupedBySession,
    showSessionViewSwitch: true,
  });

  const hasRows = page.traces.length > 0;
  const showErrorAlert = Boolean(page.error) && hasRows;
  const showFirstLoadError = Boolean(page.error) && !hasRows && !page.isFetching;
  const viewState: TracesTableViewState = showFirstLoadError
    ? 'error'
    : !page.isFetching && flags.isEmptyPageBeyondFirst
      ? 'no-more-results'
      : !page.isFetching && flags.hasNoTracesAtAll
        ? 'empty'
        : !page.isFetching && flags.hasNoSearchResults
          ? 'no-results'
          : 'ready';
  const getErrorDescription = useMemo(() => makeTracesV4ErrorDescription(intl), [intl]);
  // The cursor API returns no total, so the footer count shows only when the run fits on one page.
  const isTotalKnown = !page.hasNext && !page.hasPrev;

  const tableViewProps: TracesTableViewProps = {
    viewState,
    traces: page.traces,
    visibleColumns,
    extraColumns: assessmentColumnDefs,
    columnOrder: tableColumnOrder,
    initialColumnSizing: evalRunColumnSizing,
    onColumnSizingSettled: columnSizing.setColumnSizing,
    isLoading: page.isFetching,
    isFetching: page.isFetching,
    skeletonRowCount: url.pageSize,
    onTraceSelected: handleTraceSelected,
    getTraceHref,
    selectedTraceId: controller.activeTraceId,
    selectedForBulk: bulk.selected,
    isAllOnPageSelected: bulk.isAllVisibleChecked,
    isSomeOnPageSelected: bulk.isSomeVisibleChecked,
    onToggleBulkRow: bulk.toggle,
    onToggleBulkRows: controller.isGroupedBySession && (page.hasNext || page.hasPrev) ? undefined : bulk.toggleMany,
    onToggleBulkAll: bulk.toggleAll,
    sort: url.sort,
    dir: url.dir,
    onSort: url.setSort,
    previewLineClamp: PREVIEW_LINE_CLAMP_BY_DENSITY[density],
    getSessionHref,
    onSessionSelected: handleSessionSelected,
    onFilterByTag: controller.onFilterByTag,
    onHideColumn: handleHideColumn,
    headerBottomOverlay: assessmentHeaderBottomOverlay,
    isGroupedBySession: controller.isGroupedBySession,
    sessionsMayBeIncomplete: controller.isGroupedBySession && (page.hasNext || page.hasPrev),
    onReorderColumn: reorderColumn,
    searchValue: searchInput.input,
    onSearchChange: searchInput.setInput,
    onSearchClear: searchInput.clear,
    onSearchSubmit: searchInput.submit,
    searchInputRef,
    expandSearch: true,
    searchMinimumWidth: unifiedSearch.searchMinimumWidth,
    searchControl: unifiedSearch.renderSearchControl({
      renderFilterEditor: toolbarSlots.renderFilterEditor,
      traces: page.traces,
      inputRef: searchInputRef,
      resetKey: runUuid,
    }),
    leftControls: toolbarSlots.sessionViewSwitch,
    rightControls: toolbarSlots.rightControls,
    bannerSlot: (
      <>
        {actions.runJudges?.JudgesStatusBanner}
        {controller.selectedTraceValidation.error ? (
          <EvalRunSelectedTraceErrorAlert onRetry={controller.selectedTraceValidation.refetch} />
        ) : null}
        {assessmentSummaries.error && (
          <TracesErrorAlert
            error={assessmentSummaries.error}
            onRetry={refetchAssessmentSummaries}
            getErrorDescription={getErrorDescription}
          />
        )}
        {showErrorAlert && (
          <TracesErrorAlert error={page.error} onRetry={page.refetch} getErrorDescription={getErrorDescription} />
        )}
      </>
    ),
    pageIndex: url.pageIndex,
    pageSize: url.pageSize,
    onPageChange: controller.goToPage,
    onPageSizeChange: url.setPageSize,
    hasNext: page.hasNext,
    hasPrev: page.hasPrev,
    hidePagination: controller.isGroupedBySession && !page.hasNext && !page.hasPrev,
    hidePageSizeSelector: controller.isGroupedBySession,
    traceCount: isTotalKnown && hasRows ? page.traces.length : undefined,
    PaginationBarWrapper: AssistantAwareActionBar,
    onClearFilters: clearFilters,
    onRetry: page.refetch,
    error: page.error,
    getErrorDescription,
    customEmptyState: viewState === 'empty' ? <EvalRunTracesEmptyState /> : undefined,
  };

  const providerProps: TracesV4TableSurfaceProviderProps = {
    modelTraceExplorer: {
      renderExportTracesToDatasetsModal: actions.renderExportTracesToDatasetsModal,
      renderAddToReviewQueueDropdown: actions.AddToReviewQueueDropdown,
      DrawerComponent: AssistantAwareDrawer,
    },
    genAITracesTable: {
      experimentId,
      getTrace: actions.getTrace,
      isGroupedBySession: controller.isGroupedBySession,
    },
  };
  const drawerProps: TracesV4TraceDrawerProps = {
    traceId: controller.activeTraceId,
    onClose: closeDrawer,
    experimentId,
    traces: page.traces,
    onSelectTrace: url.setTraceId,
    runJudgeConfiguration: actions.runJudges?.runJudgeConfiguration,
  };

  return (
    <TracesV4TableSurface
      providerProps={providerProps}
      tableViewProps={tableViewProps}
      drawerProps={drawerProps}
      layoutProps={{
        paddingTop: 0,
        paddingLeft: 0,
        marginBottom: -theme.spacing.sm,
      }}
      overlaySlots={{
        deleteModal: (
          <TracesV4DeleteModal
            open={deleteOpen}
            count={pendingDeleteIds.length}
            isLoading={deleteTracesMutation.isLoading}
            error={deleteTracesMutation.error instanceof Error ? deleteTracesMutation.error : undefined}
            onConfirm={confirmDelete}
            onCancel={cancelDelete}
          />
        ),
        actionModals: (
          <>
            {actions.runJudges?.RunJudgesModal}
            {actions.editTags.EditTagsModal}
          </>
        ),
        notificationContainer,
      }}
    />
  );
};

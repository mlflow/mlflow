import { useCallback, useMemo, useRef, useState } from 'react';
import { type InputRef, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import {
  createTraceV4LongIdentifier,
  ModelTraceExplorerContextProvider,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import {
  doesTraceSupportV4API,
  GenAITracesTableProvider,
  MLFLOW_SOURCE_RUN_KEY,
  RunName,
} from '@databricks/web-shared/genai-traces-table';
import {
  EMPTY_FILTER_MODEL,
  TRACE_COLUMN_IDS,
  TracesErrorAlert,
  TracesTableView,
  type SessionHrefGetter,
  type TraceColumnId,
  type TraceHrefGetter,
  type TracesTableViewState,
} from '@databricks/web-shared/traces-table';
import { useDeleteTracesMutation } from '@mlflow/mlflow/src/experiment-tracking/components/evaluations/hooks/useDeleteTraces';
import { AssistantAwareDrawer } from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';
import { AssistantAwareActionBar } from '@mlflow/mlflow/src/common/components/AssistantAwareActionBar';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { SELECTED_TRACE_ID_QUERY_PARAM } from '@mlflow/mlflow/src/experiment-tracking/constants';
// Reuse the generic (branding-free) "/" hotkey hook from datasets-v2.
import { useSlashFocusSearch } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets-v2/hooks/useSlashFocusSearch';
import { isAssessmentColumnId } from '../utils/assessmentColumns';
import { useTracesV4Controller } from '../hooks/useTracesV4Controller';
import { useTracesV4Density } from '../hooks/useTracesV4Density';
import { useTracesV4Notifications } from '../hooks/useTracesV4Notifications';
import { useTracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import { TracesV4TraceDrawer } from './TracesV4TraceDrawer';
import { useTracesV4ToolbarSlots } from './TracesV4Toolbar';
import { TracesV4DeleteModal } from './TracesV4DeleteModal';
import { makeTracesV4ErrorDescription } from './TracesV4States';
import { TracesV4EmptyState } from './TracesV4EmptyState';
import { IssueDetectionModal } from '../../traces-v3/IssueDetectionModal';
import { TracesV4SavedViewsButton, TracesV4SharedViewBanner, useTracesV4SavedViews } from './TracesV4SavedViews';

interface TracesV4PageContentProps {
  experimentId: string;
}

// Narrows a column id to a standard `TraceColumnId` (assessment columns are namespaced separately).
const isStandardColumnId = (id: string): id is TraceColumnId => (TRACE_COLUMN_IDS as readonly string[]).includes(id);

/**
 * Layout controller for the V4 traces tab. Owns URL/data state via `useTracesV4Controller` and feeds
 * the shared `TracesTableView`, mapping the controller's flags to a single `viewState`. The two
 * providers (ModelTraceExplorer, GenAITracesTable), the drawer, the delete modal, and notifications
 * stay MLflow-side.
 */
export const TracesV4PageContent = ({ experimentId }: TracesV4PageContentProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { notify, notificationContainer } = useTracesV4Notifications();
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);

  const controller = useTracesV4Controller({ experimentId });
  const { url, page, columns, assessments, columnSizing, traceCount, bulk, searchInput, filterModel, flags } =
    controller;
  const { density, setDensity } = useTracesV4Density(experimentId);

  // Saved views (URL-first): the hook reads/writes view tags and drives the `cols`+share-key preview
  // overlay. While a shared view is applied, its previewed columns render INSTEAD of the user's own
  // (without touching localStorage until Override); otherwise the user's own columns show.
  const savedViews = useTracesV4SavedViews({
    experimentId,
    visibleColumns: columns.visibleColumns,
    setColumns: columns.setColumns,
  });
  const effectiveVisibleColumns = savedViews.previewColumns ?? columns.visibleColumns;

  // While previewing a shared view, column toggles edit the preview (the `cols` URL param) rather
  // than the user's persisted columns — matching the banner's "changes aren't saved unless you
  // override" promise. Otherwise they write the user's own localStorage as usual. The selector
  // always reflects `effectiveVisibleColumns`, so its checkboxes match the table in both modes.
  const toggleColumn = useCallback(
    (column: TraceColumnId) => {
      if (savedViews.sharedViewActive) {
        const next = effectiveVisibleColumns.includes(column)
          ? effectiveVisibleColumns.filter((id) => id !== column)
          : [...effectiveVisibleColumns, column];
        savedViews.setPreviewColumns(next);
        return;
      }
      columns.toggleColumn(column);
    },
    [savedViews, effectiveVisibleColumns, columns],
  );

  // One "Reset to defaults" in the column selector clears both standard and assessment overrides.
  // (Only reachable when not previewing — a preview's columns come from its `cols` param.)
  const resetColumns = useCallback(() => {
    columns.resetToDefaults();
    assessments.reset();
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

  const actions = useTracesV4TraceActions(experimentId, page.traces, page.refetch);

  // The selection stores the full `ModelTraceInfoV3` per trace (keyed by id), so this is the entire
  // cross-page selection — every bulk action (judges, Genie, add-to-dataset, labeling, review queue)
  // gets its expected input regardless of which page a trace was selected on.
  const selectedTraceInfos = useMemo(() => Array.from(bulk.selected.values()), [bulk.selected]);

  const deleteTracesMutation = useDeleteTracesMutation();
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Snapshot ids when the modal opens so a refetch/clear mid-prompt can't zero the count or
  // turn confirm into a no-op (mirrors the datasets-v2 delete flow).
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);

  const openDelete = useCallback(() => {
    // Delete needs ids only; the selection Map iterates entries, so take its keys.
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
          page.refetch();
          notify.success(
            intl.formatMessage(
              {
                defaultMessage: 'Deleted {count, plural, one {# trace} other {# traces}}',
                description: 'Success toast after bulk-deleting V4 traces',
              },
              { count: ids.length },
            ),
          );
        },
        // Leave selection intact + the modal open on error so the user can retry.
        onError: (err) => notify.error(err),
      },
    );
  }, [pendingDeleteIds, deleteTracesMutation, experimentId, bulk, page, notify, intl]);

  // UC-backed V4 traces must be opened with a V4 long identifier (`trace:/<catalog.schema>/<id>`);
  // the bare hex id would fall through to the legacy GetTraceInfo path and be rejected as "Invalid
  // request id". Mirrors v1's `useActiveEvaluation` selection logic.
  const handleTraceSelected = useCallback(
    (trace: ModelTraceInfoV3) =>
      url.setTraceId(doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id),
    [url],
  );
  const closeDrawer = useCallback(() => url.setTraceId(undefined), [url]);

  const getTraceHref = useCallback<TraceHrefGetter>(
    (trace) => {
      const traceId = doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;
      return `${Routes.getExperimentPageTracesTabRoute(experimentId)}?traceId=${encodeURIComponent(traceId)}`;
    },
    [experimentId],
  );

  // The session cell's only product coupling: build the single-chat-session route (matching v1),
  // deep-linking the current trace via `?selectedTraceId` so the destination opens on that trace.
  const getSessionHref = useCallback<SessionHrefGetter>(
    ({ trace, sessionId }) => {
      const baseUrl = Routes.getExperimentPageTabSingleChatSessionRoute(experimentId, sessionId);
      return trace.trace_id
        ? `${baseUrl}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: trace.trace_id }).toString()}`
        : baseUrl;
    },
    [experimentId],
  );
  const renderRunName = useCallback(
    (trace: ModelTraceInfoV3) => {
      const runUuid = trace.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
      return runUuid ? <RunName experimentId={experimentId} runUuid={runUuid} /> : undefined;
    },
    [experimentId],
  );
  // Filter button "clear all": resets exactly what the count badge totals — the popover clauses AND
  // the URL tag filters — but leaves the search box (a separate control the badge doesn't count).
  const clearAllFilters = useCallback(() => {
    controller.setFilterModel(EMPTY_FILTER_MODEL);
    url.clearTagFilters();
  }, [controller, url]);
  // Empty-state "clear filters": the broader reset that also clears the search query, since the
  // no-results state can be caused by the search too.
  const clearFilters = useCallback(() => {
    searchInput.clear();
    clearAllFilters();
  }, [searchInput, clearAllFilters]);

  const hasRows = page.traces.length > 0;
  const showErrorAlert = Boolean(page.error) && hasRows; // refetch failed but prior rows remain
  // A fetch in flight (first load OR a reload — keepPreviousData keeps isLoading false on reload)
  // renders the skeleton, so error/empty branches wait until the fetch settles.
  const showFirstLoadError = Boolean(page.error) && !hasRows && !page.isFetching;

  const getErrorDescription = useMemo(() => makeTracesV4ErrorDescription(intl), [intl]);

  // AI issue detection: the shared modal (reused from v3) seeds itself with the current bulk
  // selection, falling back to the most-recent page of traces when nothing is selected. Completion
  // toasts are handled globally by `IssueDetectionJobNotifications` (mounted in MlflowRouter).
  const [isIssueDetectionOpen, setIsIssueDetectionOpen] = useState(false);
  const selectedTraceIds = useMemo(() => Array.from(bulk.selected.keys()), [bulk.selected]);
  const availableTraceIds = useMemo(
    () => page.traces.map((trace) => trace.trace_id).filter((id): id is string => Boolean(id)),
    [page.traces],
  );

  const toolbarSlots = useTracesV4ToolbarSlots({
    filterModel,
    onFilterChange: controller.setFilterModel,
    onClearFilters: clearAllFilters,
    activeFilterCount: controller.activeFilterCount,
    visibleColumns: effectiveVisibleColumns,
    onToggleColumn: toggleColumn,
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
    onDetectIssues: () => setIsIssueDetectionOpen(true),
    savedViewsButton: <TracesV4SavedViewsButton experimentId={experimentId} savedViews={savedViews} />,
  });

  // Map the controller's flags to a single shared `viewState`. Order mirrors the prior
  // `renderTableRegion()` precedence: first-load error, then end-of-results, then the two empty
  // states, else the table. The no-warehouse case is handled via `customEmptyState` below.
  const viewState: TracesTableViewState = showFirstLoadError
    ? 'error'
    : !page.isFetching && flags.isEmptyPageBeyondFirst
      ? 'no-more-results'
      : !page.isFetching && flags.hasNoTracesAtAll
        ? 'empty'
        : !page.isFetching && flags.hasNoSearchResults
          ? 'no-results'
          : 'ready';

  const content = (
    // The two providers expose the shared trace-action building blocks (drawer add-to-dataset
    // and the dataset modal) to the drawer and Actions menu.
    <ModelTraceExplorerContextProvider
      renderExportTracesToDatasetsModal={actions.renderExportTracesToDatasetsModal}
      DrawerComponent={AssistantAwareDrawer}
    >
      <GenAITracesTableProvider experimentId={experimentId} getTrace={actions.getTrace} isGroupedBySession={false}>
        {/* Vertical-fill column for the tab. Horizontal scroll stays inside the table (TracesTable's
            own `<Table scrollable>`), so the toolbar and pagination bar keep to the visible width
            without a second scroll container of our own (ML-68750/68769) — the toolbar instead
            collapses its control labels to fit (see TracesToolbarResponsive). The negative bottom
            margin bleeds the pinned pagination bar into the shared content wrapper's 8px bottom
            padding, aligning it with the sidebar. */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
            gap: theme.spacing.md,
            paddingTop: theme.spacing.md,
            paddingLeft: theme.spacing.md,
            marginBottom: -theme.spacing.sm,
          }}
        >
          <TracesTableView
            viewState={viewState}
            // Table
            traces={page.traces}
            visibleColumns={effectiveVisibleColumns}
            extraColumns={assessments.columnDefs}
            initialColumnSizing={columnSizing.columnSizing}
            onColumnSizingSettled={columnSizing.setColumnSizing}
            isLoading={page.isFetching}
            isFetching={page.isFetching}
            skeletonRowCount={url.pageSize}
            onTraceSelected={handleTraceSelected}
            selectedTraceId={url.traceId}
            selectedForBulk={bulk.selected}
            isAllOnPageSelected={bulk.isAllVisibleChecked}
            isSomeOnPageSelected={bulk.isSomeVisibleChecked}
            onToggleBulkRow={bulk.toggle}
            onToggleBulkAll={bulk.toggleAll}
            sort={url.sort}
            dir={url.dir}
            onSort={url.setSort}
            size={density}
            getTraceHref={getTraceHref}
            getSessionHref={getSessionHref}
            onFilterByTag={controller.onFilterByTag}
            renderRunName={renderRunName}
            onHideColumn={handleHideColumn}
            // Toolbar slots (built by useTracesV4ToolbarSlots) + banner slot
            searchValue={searchInput.input}
            onSearchChange={searchInput.setInput}
            onSearchClear={searchInput.clear}
            onSearchSubmit={searchInput.submit}
            searchInputRef={searchInputRef}
            leftControls={toolbarSlots.leftControls}
            rightControls={toolbarSlots.rightControls}
            bannerSlot={
              <>
                <TracesV4SharedViewBanner savedViews={savedViews} />
                {actions.runJudges?.JudgesStatusBanner}
                {showErrorAlert && (
                  <TracesErrorAlert
                    error={page.error}
                    onRetry={page.refetch}
                    getErrorDescription={getErrorDescription}
                  />
                )}
              </>
            }
            // Pagination
            pageIndex={url.pageIndex}
            pageSize={url.pageSize}
            onPageChange={controller.goToPage}
            onPageSizeChange={url.setPageSize}
            hasNext={page.hasNext}
            hasPrev={page.hasPrev}
            // "{n} of {total}" footer count (bottom-left).
            traceCount={traceCount.currentCount}
            traceTotal={traceCount.totalCount}
            isTraceCountLoading={traceCount.isTotalLoading}
            // Reserve the pinned pagination bar's height with the floating-obstruction store so the
            // Assistant FAB rises above it instead of overlapping the prev/next/page-size controls.
            PaginationBarWrapper={AssistantAwareActionBar}
            // States
            onClearFilters={clearFilters}
            onRetry={page.refetch}
            error={page.error}
            getErrorDescription={getErrorDescription}
            customEmptyState={
              // Short-circuits before the viewState switch (keeping toolbar + banner). Gated on the
              // resolved `empty` state, not `hasNoTracesAtAll` alone, so error / no-results /
              // no-more-results still flow through the switch — a trace-id-search miss lands on
              // `no-results`, and a first-load error on `error`, not the quickstart.
              viewState === 'empty' ? <TracesV4EmptyState experimentId={experimentId} /> : undefined
            }
          />

          <TracesV4DeleteModal
            open={deleteOpen}
            count={pendingDeleteIds.length}
            isLoading={deleteTracesMutation.isLoading}
            error={deleteTracesMutation.error instanceof Error ? deleteTracesMutation.error : undefined}
            onConfirm={confirmDelete}
            onCancel={cancelDelete}
          />

          <TracesV4TraceDrawer
            traceId={url.traceId}
            onClose={closeDrawer}
            experimentId={experimentId}
            traces={page.traces}
            onSelectTrace={url.setTraceId}
            runJudgeConfiguration={actions.runJudges?.runJudgeConfiguration}
          />

          {actions.runJudges?.RunJudgesModal}
          {actions.editTags.EditTagsModal}

          {isIssueDetectionOpen && (
            <IssueDetectionModal
              key={experimentId}
              onClose={() => setIsIssueDetectionOpen(false)}
              experimentId={experimentId}
              initialSelectedTraceIds={selectedTraceIds}
              availableTraceIds={availableTraceIds}
            />
          )}

          {notificationContainer}
        </div>
      </GenAITracesTableProvider>
    </ModelTraceExplorerContextProvider>
  );

  return content;
};

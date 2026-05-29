import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, DangerModal, type InputRef, useDesignSystemTheme } from '@databricks/design-system';
import { Global } from '@emotion/react';
import { ResizableBox } from 'react-resizable';
import { FormattedMessage, useIntl } from 'react-intl';
import { useLocalStorage } from '@databricks/web-shared/hooks';
// useNavigationBlock stubbed for OSS — see useDatasetsPageQuery comment in PR2 plan
import type { Dataset, DatasetRecord } from '../hooks/useDatasetsQueries';
import { useDeleteDatasetRecordsMutation } from '../hooks/useDatasetsQueries';
import { useDatasetRecordsController } from '../hooks/useDatasetRecordsController';
import type { PendingNewRecord } from '../hooks/useRecordCreateState';
import { useDatasetNotifications } from '../hooks/useDatasetNotifications';
import { useGuardedTransition } from '../hooks/useGuardedTransition';
import { useSlashFocusSearch } from '../hooks/useSlashFocusSearch';
import { DatasetsBreadcrumbs } from './DatasetsBreadcrumbs';
import { DatasetRecordsToolbar } from './DatasetRecordsToolbar';
import { DatasetRecordsTable } from './DatasetRecordsTable';
import { DatasetRecordsEmptyState, DatasetRecordsNoResultsEmptyState } from './DatasetRecordsEmptyState';
import { DatasetRecordsLoadingSkeleton } from './DatasetRecordsLoadingSkeleton';
import { DatasetRecordSidePanel } from './DatasetRecordSidePanel';
import { DatasetRecordsColumnSelector } from './DatasetRecordsColumnSelector';
import { DatasetRecordsCount } from './DatasetRecordsCount';
import { BulkDeleteRecordsModal } from './BulkDeleteRecordsModal';
import { DatasetDetailKebabMenu } from './DatasetDetailKebabMenu';
import { SidePanelResizeHandle } from './SidePanelResizeHandle';
import { TraceModal } from './TraceModal';
import { DEFAULT_RECORD_PAGE_SIZE } from '../utils/constants';

/**
 * OSS implementation of `useNavigationBlock` — what universe gets from
 * `@databricks/web-shared/routing`. Covers the tab-close / refresh case via `beforeunload`,
 * which is the most common "I lost my work" scenario.
 *
 * Limitation: in-app navigation (clicking another row, breadcrumb, side-nav) is NOT
 * intercepted. react-router v6.7+ added `useBlocker` for that; OSS is on 6.4.1 today, so
 * we'd need a router bump first. Universe's `block(handler)` callback receives a `retry`
 * fn for the modal flow — our stub returns a no-op handler since `beforeunload` resolves
 * synchronously through the browser-native confirm dialog instead.
 */
const useNavigationBlock = () => {
  const activeRef = useRef(false);
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!activeRef.current) return undefined;
      // Modern browsers display their own generic message, but Chrome/Firefox still
      // require `preventDefault` + `returnValue` to opt into the prompt.
      e.preventDefault();
      e.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);
  return (_handler: (tx: { retry: () => void }) => void) => {
    activeRef.current = true;
    return () => {
      activeRef.current = false;
    };
  };
};

const SIDE_PANEL_DEFAULT_WIDTH = 640;
const SIDE_PANEL_MIN_WIDTH = 400;
const SIDE_PANEL_MAX_WIDTH = 1000;
const SIDE_PANEL_WIDTH_STORAGE_KEY = 'mlflow.eval-datasets-v2.side-panel-width';
const SIDE_PANEL_WIDTH_STORAGE_VERSION = 1;

export interface DatasetDetailPageContentProps {
  experimentId: string;
  datasetId: string;
  /**
   * Loaded dataset metadata. The parent page (`ExperimentEvaluationDatasetDetailPage`)
   * handles loading/404 states, so this is always a fully-resolved `Dataset` here.
   */
  dataset: Dataset;
}

export const DatasetDetailPageContent = ({ experimentId, datasetId, dataset }: DatasetDetailPageContentProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { notify, notificationContainer } = useDatasetNotifications();
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);

  const { url, records, selectedRecord, bulk, columns, flags, searchInput, setPageIndex, columnWidths } =
    useDatasetRecordsController({ experimentId, datasetId });

  const deleteRecordsMutation = useDeleteDatasetRecordsMutation(datasetId);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  // Snapshot of ids captured when the confirm modal opens. The modal renders `count` and the
  // confirm handler acts on this snapshot — never on the live `bulk.selected` — so a refetch,
  // page clamp, or any future code path that triggers `bulk.clear()` mid-prompt cannot zero
  // out the count or turn the confirm click into a no-op.
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);
  // Live preview of the in-progress new record. Non-null while the side panel is open in
  // create mode; the records table renders a synthetic row at the top reflecting these
  // values so the create flow feels continuous with the table instead of modal-disjoint.
  // Raw text fields let the row mirror what the user is typing even before the JSON parses.
  const [pendingNewRecord, setPendingNewRecord] = useState<PendingNewRecord | null>(null);

  // Trace explorer modal — opened from the side panel's Source row when a record is
  // trace-sourced. Hosting it here (instead of inside the side panel) lets the modal
  // overlay the entire page.
  const [isTraceModalOpen, setIsTraceModalOpen] = useState(false);
  const [selectedTraceId, setSelectedTraceId] = useState<string>('');

  // Side panel width persists globally (one preference across all datasets / experiments),
  // since it expresses the user's preferred layout — not anything dataset-specific. React
  // state is updated only on drag stop; during a drag we mutate the table wrapper's
  // `minWidth` directly via `tableWrapperRef` so the heavy children (JSON editors, table
  // rows) don't re-render on every pointer tick. `ResizableBox` already drives the box's
  // own visual width via its internal state, so the panel itself still resizes live.
  const [persistedPanelWidth, setPersistedPanelWidth] = useLocalStorage<number>({
    key: SIDE_PANEL_WIDTH_STORAGE_KEY,
    version: SIDE_PANEL_WIDTH_STORAGE_VERSION,
    initialValue: SIDE_PANEL_DEFAULT_WIDTH,
  });
  // Clamp on read so a stale localStorage value outside the current constraints can't break
  // the layout — the user just gets a width inside the allowed range until they resize again.
  const [panelWidth, setPanelWidth] = useState(() =>
    Math.min(SIDE_PANEL_MAX_WIDTH, Math.max(SIDE_PANEL_MIN_WIDTH, persistedPanelWidth)),
  );
  // Suppress page-level text selection while dragging the divider — react-resizable captures
  // pointer events on the handle itself, but a fast drag past the handle can still highlight
  // content underneath without this global override.
  const [isResizing, setIsResizing] = useState(false);
  // Live drag target for the table wrapper's `minWidth`. The drag handler mutates this
  // element's style directly (see the ResizableBox below); committing to React state
  // happens once on resize stop.
  const tableWrapperRef = useRef<HTMLDivElement | null>(null);

  const panelMode: 'edit' | 'create' | null = url.recordId ? 'edit' : pendingNewRecord !== null ? 'create' : null;
  const panelOpen = panelMode !== null;

  // Dirty flag is owned by the side panel (drives its editor state) but consumed here so
  // every transition that would discard those edits — close, record-switch, mode-switch,
  // in-app navigation — funnels through one `useGuardedTransition` and one DangerModal.
  const [isPanelDirty, setIsPanelDirty] = useState(false);
  const {
    isPromptOpen,
    requestTransition,
    confirm: confirmTransition,
    cancel: cancelTransition,
  } = useGuardedTransition({ isDirty: isPanelDirty });

  // Block in-app navigation (breadcrumb, back, future links) while dirty and route the
  // confirm through the same modal. Must use the {retry} contract — the Databricks blocker
  // discards the return value, so returning `window.confirm()` (the previous Prompt impl)
  // silently dropped navigation on the floor after the user pressed OK.
  const block = useNavigationBlock();
  // Read inside the blocker without forcing a re-register on every isPromptOpen flip.
  const isPromptOpenRef = useRef(isPromptOpen);
  useEffect(() => {
    isPromptOpenRef.current = isPromptOpen;
  }, [isPromptOpen]);
  useEffect(() => {
    if (!isPanelDirty) return undefined;
    const unblock = block(({ retry }) => {
      // When the modal is already open, the user just clicked Discard and the stashed
      // transition's URL change (handleClosePanel / handleRecordSelected / handleAddRecord)
      // is hitting this same blocker. Re-routing through requestTransition would overwrite
      // the in-flight transition; the subsequent setPendingTransition(null) inside confirm()
      // would then drop the navigation entirely — modal closes, panel stays open. Pass the
      // navigation straight through instead.
      if (isPromptOpenRef.current) {
        unblock();
        retry();
        return;
      }
      requestTransition(() => {
        unblock();
        retry();
      });
    });
    return unblock;
  }, [isPanelDirty, block, requestTransition]);

  // The panel reports dirty state via `onDirtyChange`, but an unmount-only cleanup can't
  // fire that callback (the effect re-runs would flicker false→true). Reset here so closing
  // the panel doesn't leave the navigation block armed with nothing to guard.
  useEffect(() => {
    if (!panelOpen) setIsPanelDirty(false);
  }, [panelOpen]);

  const handleOpenBulkDelete = useCallback(() => {
    setPendingDeleteIds(Array.from(bulk.selected));
    setBulkDeleteOpen(true);
  }, [bulk.selected]);

  const handleCancelBulkDelete = useCallback(() => {
    setBulkDeleteOpen(false);
    setPendingDeleteIds([]);
  }, []);

  const handleConfirmBulkDelete = useCallback(() => {
    if (pendingDeleteIds.length === 0) return;
    const ids = pendingDeleteIds;
    deleteRecordsMutation.mutate(ids, {
      onSuccess: () => {
        bulk.clear();
        setBulkDeleteOpen(false);
        setPendingDeleteIds([]);
        // The mutation's onSettled invalidates ['listDatasetRecords', datasetId]; no manual
        // refetch — a second one races with the invalidation and can un-delete rows on slow
        // networks if the explicit refetch lands first.
        notify.success(
          intl.formatMessage(
            {
              defaultMessage: 'Deleted {count, plural, one {# record} other {# records}}',
              description: 'Success toast after bulk-deleting V2 dataset records',
            },
            { count: ids.length },
          ),
        );
      },
      onError: (err) => notify.error(err),
    });
  }, [pendingDeleteIds, deleteRecordsMutation, bulk, intl, notify]);

  const handleAddRecord = useCallback(() => {
    // Open the panel in create mode by seeding the pending-new-record preview. Clearing
    // `recordId` ensures the panel's mode derivation lands on 'create' even if an edit was
    // previously open. The fake row appears at the top of the table immediately.
    requestTransition(() => {
      setPendingNewRecord({
        inputsText: '',
        expectationsText: '',
        inputs: undefined,
        expectations: undefined,
        tags: {},
      });
      url.setRecordId(undefined);
    });
  }, [requestTransition, url]);

  const handleRecordSelected = useCallback(
    (record: DatasetRecord) => {
      requestTransition(() => {
        setPendingNewRecord(null);
        url.setRecordId(record.dataset_record_id);
      });
    },
    [requestTransition, url],
  );

  const handleClosePanel = useCallback(() => {
    requestTransition(() => {
      url.setRecordId(undefined);
      setPendingNewRecord(null);
    });
  }, [requestTransition, url]);

  const handleOpenTraceModal = useCallback((traceId: string) => {
    setSelectedTraceId(traceId);
    setIsTraceModalOpen(true);
  }, []);

  const handleEditSaveSuccess = useCallback(
    () =>
      notify.success(
        intl.formatMessage({
          defaultMessage: 'Record saved',
          description: 'Success toast after saving an edited dataset record',
        }),
      ),
    [intl, notify],
  );

  const handleCreateSaveSuccess = useCallback(() => {
    setPendingNewRecord(null);
    records.refetch();
    notify.success(
      intl.formatMessage({
        defaultMessage: 'Record added',
        description: 'Success toast after adding a new V2 dataset record',
      }),
    );
  }, [intl, notify, records]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        // Outer wrappers (PageWrapper + ExperimentPageTabs) already contribute spacing
        // on the right (24px) and bottom (8px); only top and left need padding here.
        paddingTop: theme.spacing.md,
        paddingLeft: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <div
          css={{
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            gap: theme.spacing.md,
            // Right padding when the side panel is open separates the kebab (in
            // breadcrumbs) and the Add-record button (in the toolbar) from the
            // panel. Matches the panel's own internal `lg` horizontal padding so
            // the visual gutter is symmetric on both sides of the panel border.
            paddingRight: panelOpen ? theme.spacing.lg : 0,
          }}
        >
          <DatasetsBreadcrumbs
            experimentId={experimentId}
            datasetName={dataset.name}
            rightActions={<DatasetDetailKebabMenu experimentId={experimentId} dataset={dataset} notify={notify} />}
          />

          {records.error instanceof Error && (
            <Alert
              componentId="mlflow.eval-datasets-v2.records.fetch-error"
              type="error"
              message={records.error.message}
              closable={false}
            />
          )}

          {records.isLoading ? (
            // First load: show a full-area skeleton instead of toolbar + in-table skeleton, so
            // empty datasets resolve directly into the empty state without the toolbar briefly
            // flashing in. Subsequent refetches (search, refresh) skip this branch and keep the
            // toolbar visible — the records table's own `isFetching` path handles those.
            <DatasetRecordsLoadingSkeleton />
          ) : flags.hasNoRecordsAtAll && pendingNewRecord === null ? (
            // Hide the empty state once the user clicks "Add record" on an empty dataset so
            // the toolbar + table render and the phantom row preview appears alongside the
            // open side panel — same continuous-create flow as the non-empty case.
            <DatasetRecordsEmptyState onAddRecord={handleAddRecord} />
          ) : (
            <>
              <DatasetRecordsToolbar
                searchInputValue={searchInput.input}
                onSearchInputChange={searchInput.setInput}
                onSearchClear={searchInput.clear}
                onRefresh={records.refetch}
                isRefreshing={records.isFetching && !records.isLoading}
                lastRefreshTime={records.dataUpdatedAt > 0 ? records.dataUpdatedAt : undefined}
                onAddRecord={handleAddRecord}
                searchInputRef={searchInputRef}
                trailingControls={
                  panelOpen ? null : (
                    <>
                      <DatasetRecordsColumnSelector
                        visibleColumns={columns.visibleColumns}
                        onToggleColumn={columns.toggleColumn}
                        onResetToDefaults={columns.resetToDefaults}
                      />
                      {records.allRecords.length > 0 && (
                        // 1px nudge on top of the toolbar's `sm` gap to give the count
                        // a hair more breathing room from the Columns button.
                        <span css={{ marginLeft: 1 }}>
                          <DatasetRecordsCount
                            filtered={records.totalRecords}
                            total={records.allRecords.length}
                            hasActiveSearch={flags.hasActiveSearch}
                          />
                        </span>
                      )}
                    </>
                  )
                }
                selectionCount={bulk.selected.size}
                onBulkDelete={handleOpenBulkDelete}
                onBulkClear={bulk.clear}
              />
              {flags.hasNoSearchResults ? (
                <DatasetRecordsNoResultsEmptyState searchQuery={url.search} onClearSearch={() => url.setSearch('')} />
              ) : (
                // When the side panel is open, the outer left container shrinks by exactly
                // `panelWidth`. The inner div's `minWidth: calc(100% + panelWidth)` keeps the
                // table at the width it had before the panel opened, forcing the outer wrapper
                // to scroll horizontally instead of compressing column widths. During a drag
                // the value is updated via direct style mutation on `tableWrapperRef` (see the
                // ResizableBox below) so the table re-flows live without re-rendering this
                // subtree on every pointer tick; the React-rendered value catches up on stop.
                // The `minWidth` lives on `style` rather than `css` so that on close React's
                // style diff actively removes the inline `min-width` from the element. An
                // Emotion class would just stop generating the rule, but the inline value
                // written by the live drag handler would linger and leave the table wider
                // than the viewport after the panel goes away.
                <div css={{ flex: 1, minHeight: 0, overflowX: 'auto' }}>
                  <div
                    ref={tableWrapperRef}
                    style={panelOpen ? { minWidth: `calc(100% + ${panelWidth}px)` } : undefined}
                  >
                    <DatasetRecordsTable
                      records={records.records}
                      totalRecords={records.totalRecords}
                      pageIndex={url.pageIndex}
                      pageSize={DEFAULT_RECORD_PAGE_SIZE}
                      onPageChange={setPageIndex}
                      isLoading={records.isLoading}
                      isFetching={records.isFetching}
                      onRecordSelected={handleRecordSelected}
                      selectedRecordId={url.recordId}
                      visibleColumns={columns.visibleColumns}
                      columnSizing={columnWidths.columnSizing}
                      setColumnSizing={columnWidths.setColumnSizing}
                      selectedForBulk={bulk.selected}
                      isAllOnPageSelected={bulk.isAllVisibleChecked}
                      isSomeOnPageSelected={bulk.isSomeVisibleChecked}
                      onToggleBulkRow={bulk.toggle}
                      onToggleBulkAll={bulk.toggleAll}
                      sort={url.sort}
                      dir={url.dir}
                      onSort={url.setSort}
                      pendingNewRecord={pendingNewRecord}
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
        {panelOpen && (
          <ResizableBox
            // Outer wrapper holds the persistent layout + negative right-margin trick (so the
            // panel border reaches the viewport edge despite PageWrapper + ExperimentPageTabs
            // padding). Putting these on the ResizableBox itself — not a child — is required
            // because ResizableBox owns the width/flex slot in the row.
            css={{
              flexShrink: 0,
              borderLeft: `1px solid ${theme.colors.border}`,
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              marginRight: -theme.spacing.lg,
              position: 'relative',
            }}
            width={panelWidth}
            axis="x"
            resizeHandles={['w']}
            minConstraints={[SIDE_PANEL_MIN_WIDTH, 0]}
            maxConstraints={[SIDE_PANEL_MAX_WIDTH, 0]}
            onResize={(_event, { size }) => {
              // Bypass React entirely during drag. `ResizableBox` already updates the
              // panel's own visual width via its internal state, so the only sibling that
              // needs the live value is the table wrapper's `minWidth`. Mutating it
              // directly keeps `DatasetDetailPageContent` (and its heavy children — JSON
              // editors, per-row `JsonPreviewCell`s) from re-rendering on every tick.
              if (tableWrapperRef.current) {
                tableWrapperRef.current.style.minWidth = `calc(100% + ${size.width}px)`;
              }
            }}
            onResizeStart={() => setIsResizing(true)}
            onResizeStop={(_event, { size }) => {
              setIsResizing(false);
              // Commit the final width to React state so the next render of the table
              // wrapper produces the same `minWidth` we just mutated (otherwise an
              // unrelated re-render later would snap back to the pre-drag value), and
              // persist it. Both setters batch into a single re-render under React 18.
              setPanelWidth(size.width);
              setPersistedPanelWidth(size.width);
            }}
            handle={<SidePanelResizeHandle />}
          >
            <DatasetRecordSidePanel
              // Remount the side panel — including the Monaco editors — on every
              // create↔edit mode flip. Both `useRecordSaveState` and `useRecordCreateState`
              // run in the same instance to satisfy rules-of-hooks; without a key, the
              // edit-mode editor's text drifts to '' during a create-mode interlude (its
              // `record` becomes undefined, the recordId-change effect resets it), and the
              // subsequent reset to the next selected row's text races with Monaco's
              // change-tracking plugin. Keying on mode gives the next mode a fresh hook
              // instance initialized synchronously with the correct content.
              key={panelMode ?? 'edit'}
              mode={panelMode ?? 'edit'}
              datasetId={datasetId}
              record={selectedRecord}
              existingRecords={records.allRecords}
              open={panelOpen}
              onClose={handleClosePanel}
              onSaveSuccess={panelMode === 'create' ? handleCreateSaveSuccess : handleEditSaveSuccess}
              onSaveError={notify.error}
              onPendingChange={setPendingNewRecord}
              onDirtyChange={setIsPanelDirty}
              onOpenTraceModal={handleOpenTraceModal}
            />
          </ResizableBox>
        )}
        {isResizing && <Global styles={{ 'body, :host': { userSelect: 'none' } }} />}
      </div>

      <BulkDeleteRecordsModal
        open={bulkDeleteOpen}
        count={pendingDeleteIds.length}
        isLoading={deleteRecordsMutation.isLoading}
        error={deleteRecordsMutation.error instanceof Error ? deleteRecordsMutation.error : undefined}
        onConfirm={handleConfirmBulkDelete}
        onCancel={handleCancelBulkDelete}
      />

      <DangerModal
        componentId="mlflow.eval-datasets-v2.side-panel.discard-prompt"
        visible={isPromptOpen}
        title={
          <FormattedMessage
            defaultMessage="Discard unsaved changes?"
            description="Title of the prompt shown when leaving the dataset record side panel with unsaved edits"
          />
        }
        okText={intl.formatMessage({
          defaultMessage: 'Discard',
          description: 'Confirm-button text for the discard-unsaved-changes prompt',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Keep editing',
          description: 'Cancel-button text for the discard-unsaved-changes prompt',
        })}
        onOk={confirmTransition}
        onCancel={cancelTransition}
      >
        <FormattedMessage
          defaultMessage="Your changes will be lost if you leave this panel without saving."
          description="Body text for the discard-unsaved-changes prompt on the dataset record side panel"
        />
      </DangerModal>

      <TraceModal visible={isTraceModalOpen} onClose={() => setIsTraceModalOpen(false)} traceId={selectedTraceId} />

      {notificationContainer}
    </div>
  );
};

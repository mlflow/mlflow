import { type RefObject } from 'react';
import { type InputRef, useDesignSystemTheme } from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import type { ColumnSizingState } from '@tanstack/react-table';
import type {
  SessionHrefGetter,
  SessionSelectionHandler,
  PageSize,
  SortDirection,
  TraceColumnId,
  TraceHrefGetter,
  TraceTableColumn,
} from './types';
import { TracesTable } from './TracesTable';
import { TracesTableToolbar } from './TracesTableToolbar';
import { TracesPaginationBar } from './TracesPaginationBar';
import {
  type ErrorDescriptionGetter,
  TracesEmptyState,
  TracesErrorState,
  TracesNoMoreResultsState,
  TracesNoResultsState,
} from './TracesTableStates';

/**
 * The single region state the consumer computes and passes in. Kept consumer-computed on purpose —
 * it keeps warehouse/monitoring/data logic out of the shared layer.
 * - `loading`     first load, no prior rows → skeleton
 * - `ready`       rows present (or a background refetch keeping prior rows) → table + pagination
 * - `empty`       no traces at all → empty state
 * - `no-results`  filters/search matched nothing → no-results state (offers clear)
 * - `no-more-results` paged one step past the last page → end state, pagination kept
 * - `error`       first-load error (no prior rows) → error state (offers retry)
 */
export type TracesTableViewState = 'loading' | 'ready' | 'empty' | 'no-results' | 'no-more-results' | 'error';

export interface TracesTableViewProps {
  viewState: TracesTableViewState;

  // Table passthrough (see TracesTableProps).
  traces: ModelTraceInfoV3[];
  visibleColumns: TraceColumnId[];
  extraColumns?: TraceTableColumn[];
  initialColumnSizing: ColumnSizingState;
  onColumnSizingSettled: (sizing: ColumnSizingState) => void;
  isLoading: boolean;
  isFetching: boolean;
  skeletonRowCount: number;
  onTraceSelected: (trace: ModelTraceInfoV3) => void;
  selectedTraceId?: string;
  selectedForBulk: ReadonlyMap<string, ModelTraceInfoV3>;
  isAllOnPageSelected: boolean;
  isSomeOnPageSelected: boolean;
  onToggleBulkRow: (trace: ModelTraceInfoV3) => void;
  /** Toggle-select every trace in a session header row; omit to disable session-level selection. */
  onToggleBulkRows?: (traces: ModelTraceInfoV3[]) => void;
  onToggleBulkAll: () => void;
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  getTraceHref?: TraceHrefGetter;
  getSessionHref?: SessionHrefGetter;
  /** Handles clicks on a grouped session summary row — forwarded to the table. */
  onSessionSelected?: SessionSelectionHandler;
  onFilterByTag?: (key: string, value: string) => void;
  renderRunName?: (trace: ModelTraceInfoV3) => React.ReactNode;
  /** Hides the column with the given id — forwarded to the table's per-header menu. */
  onHideColumn: (columnId: string) => void;
  /** Groups traces with a session id into collapsible session rows — forwarded to the table. */
  isGroupedBySession?: boolean;
  /** Row-height density forwarded to the table (`'small'` = compact rows). Defaults to `'default'`. */
  size?: 'default' | 'small';

  // Toolbar passthrough.
  searchValue: string;
  onSearchChange: (next: string) => void;
  onSearchClear: () => void;
  searchInputRef?: RefObject<InputRef>;
  searchPlaceholder?: string;
  searchSuffix?: React.ReactNode;
  onSearchSubmit?: () => void;
  leftControls?: React.ReactNode;
  rightControls?: React.ReactNode;
  /** Rendered between the toolbar and the table region (e.g. a refetch-error alert or status banner). */
  bannerSlot?: React.ReactNode;

  // Pagination passthrough.
  pageIndex: number;
  pageSize: PageSize;
  onPageChange: (pageIndex: number) => void;
  onPageSizeChange: (pageSize: PageSize) => void;
  hasNext: boolean;
  hasPrev: boolean;
  /** Optional "{n} of {total}" footer count (see `TracesPaginationBar`). */
  traceCount?: number;
  traceTotal?: number;
  isTraceCountLoading?: boolean;
  /** Hide the whole pagination bar — grouped-by-session mode when the one big page holds everything. */
  hidePagination?: boolean;
  /** Hide just the page-size selector — grouped mode fetches one large page, so it's moot. */
  hidePageSizeSelector?: boolean;

  // State handling.
  /** Clear active search/filters — wired to the no-results state's clear affordance. */
  onClearFilters: () => void;
  onRetry: () => void;
  error?: unknown;
  getErrorDescription?: ErrorDescriptionGetter;
  /**
   * Short-circuit region content, rendered instead of the `viewState` switch when provided. Use for
   * a product-specific pre-query state (e.g. MLflow's "select a SQL warehouse"). The toolbar and
   * `bannerSlot` still render above it.
   */
  customEmptyState?: React.ReactNode;
  /**
   * Optional wrapper around the bottom pagination bar. Load-bearing but optional (like
   * `getSessionHref`): omit it and the bar renders bare. MLflow passes `AssistantAwareActionBar` so
   * the floating Assistant button rises above the pinned bar instead of overlapping its controls.
   * The wrapper must be layout-neutral (render its children as-is); it only measures/reserves space.
   */
  PaginationBarWrapper?: React.ComponentType<{ children: React.ReactNode }>;
}

/**
 * Fully-controlled convenience wrapper composing toolbar + table + pagination + states. It always
 * renders the toolbar (and `bannerSlot`), short-circuits to `customEmptyState` when provided, then
 * `switch`es on the consumer-computed `viewState` to render the table+pagination or the appropriate
 * state. It owns no data, URL, or product logic — every input is a prop.
 */
export const TracesTableView: React.FC<TracesTableViewProps> = (props: TracesTableViewProps) => {
  const { theme } = useDesignSystemTheme();
  const {
    viewState,
    leftControls,
    rightControls,
    bannerSlot,
    customEmptyState,
    searchValue,
    onSearchChange,
    onSearchClear,
    searchInputRef,
    searchPlaceholder,
    searchSuffix,
    onSearchSubmit,
    onClearFilters,
    onRetry,
    error,
    getErrorDescription,
    pageIndex,
    pageSize,
    onPageChange,
    onPageSizeChange,
    hasNext,
    hasPrev,
  } = props;

  const { PaginationBarWrapper } = props;
  const paginationBarInner = (
    <TracesPaginationBar
      pageIndex={pageIndex}
      pageSize={pageSize}
      onPageChange={onPageChange}
      onPageSizeChange={onPageSizeChange}
      hasNext={hasNext}
      hasPrev={hasPrev}
      count={props.traceCount}
      total={props.traceTotal}
      isCountLoading={props.isTraceCountLoading}
      hidePageSizeSelector={props.hidePageSizeSelector}
    />
  );
  // Wrap the bar in the consumer's obstruction-aware shell when provided (else render it bare). When
  // `hidePagination` is set (grouped mode with a single big page), drop the bar entirely.
  const paginationBar = props.hidePagination ? null : PaginationBarWrapper ? (
    <PaginationBarWrapper>{paginationBarInner}</PaginationBarWrapper>
  ) : (
    paginationBarInner
  );

  const table = (
    <TracesTable
      traces={props.traces}
      visibleColumns={props.visibleColumns}
      extraColumns={props.extraColumns}
      initialColumnSizing={props.initialColumnSizing}
      onColumnSizingSettled={props.onColumnSizingSettled}
      isLoading={props.isLoading}
      isFetching={props.isFetching}
      skeletonRowCount={props.skeletonRowCount}
      onTraceSelected={props.onTraceSelected}
      selectedTraceId={props.selectedTraceId}
      selectedForBulk={props.selectedForBulk}
      isAllOnPageSelected={props.isAllOnPageSelected}
      isSomeOnPageSelected={props.isSomeOnPageSelected}
      onToggleBulkRow={props.onToggleBulkRow}
      onToggleBulkRows={props.onToggleBulkRows}
      onToggleBulkAll={props.onToggleBulkAll}
      sort={props.sort}
      dir={props.dir}
      onSort={props.onSort}
      getTraceHref={props.getTraceHref}
      getSessionHref={props.getSessionHref}
      onSessionSelected={props.onSessionSelected}
      onFilterByTag={props.onFilterByTag}
      renderRunName={props.renderRunName}
      onHideColumn={props.onHideColumn}
      isGroupedBySession={props.isGroupedBySession}
      size={props.size}
    />
  );

  const renderRegion = () => {
    if (customEmptyState) {
      return customEmptyState;
    }
    switch (viewState) {
      case 'empty':
        return <TracesEmptyState />;
      case 'no-results':
        return <TracesNoResultsState onClearFilters={onClearFilters} />;
      case 'error':
        return <TracesErrorState onRetry={onRetry} error={error} getErrorDescription={getErrorDescription} />;
      case 'no-more-results':
        // Keep the pagination bar so the user can step back to the last page with rows.
        return (
          <>
            <TracesNoMoreResultsState onPrevious={() => onPageChange(pageIndex - 1)} />
            {paginationBar}
          </>
        );
      case 'loading':
      case 'ready':
      default:
        return (
          <>
            {table}
            {paginationBar}
          </>
        );
    }
  };

  return (
    <>
      <TracesTableToolbar
        searchValue={searchValue}
        onSearchChange={onSearchChange}
        onSearchClear={onSearchClear}
        searchInputRef={searchInputRef}
        searchPlaceholder={searchPlaceholder}
        searchSuffix={searchSuffix}
        onSearchSubmit={onSearchSubmit}
        leftControls={leftControls}
        rightControls={rightControls}
      />
      {bannerSlot}
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, gap: theme.spacing.sm }}>
        {renderRegion()}
      </div>
    </>
  );
};

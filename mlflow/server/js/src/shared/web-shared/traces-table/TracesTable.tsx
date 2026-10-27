import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  DragIcon,
  SpeechBubbleIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { CollisionDetection, DragEndEvent, DragStartEvent, Modifier, Modifiers } from '@dnd-kit/core';
import { closestCenter, DndContext, DragOverlay, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { snapCenterToCursor } from '@dnd-kit/modifiers';
import { horizontalListSortingStrategy, SortableContext, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useIntl } from '@databricks/i18n';
import type { CSSObject } from '@emotion/react';
// This react-table package predates `useReactTableWithDeepMemo`. Use the canonical
// `useReactTable_unverifiedWithReact18` (the same hook `GenAiTracesTableBody` uses for react-query
// data): `data` comes reference-stable from `useTracesPageQuery`, and `columns`/`meta` are memoized
// below, so no deep-equality guard is needed.
import { useReactTable_unverifiedWithReact18 } from '../react-table/useReactTable';
import { createTraceV4LongIdentifier } from '../model-trace-explorer/ModelTraceExplorer.utils';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { SESSION_ID_METADATA_KEY } from '../model-trace-explorer/constants';
import { doesTraceSupportV4API } from '../genai-traces-table/utils/TraceLocationUtils';
import {
  getTraceInfoInputPreview,
  getTraceInfoInputs,
  getTraceInfoOutputs,
} from '../genai-traces-table/utils/TraceUtils';
import { Link } from '../genai-traces-table/utils/RoutingUtils';
import { type ColumnSizingState, flexRender, getCoreRowModel, type Row } from '@tanstack/react-table';
import { Fragment, memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { createPath } from 'react-router';
import { createPortal } from 'react-dom';
import { isComparisonSortableTraceColumn, isSortableTraceColumn } from './constants';
import type {
  SessionHrefGetter,
  SessionSelectionHandler,
  SortDirection,
  TraceColumnId,
  TraceColumnHeaderAction,
  TraceColumnHeaderSummary,
  SessionComparisonRow,
  TraceComparisonRow,
  TraceHrefGetter,
  TraceSortColumnId,
  TraceTableColumn,
  TracesComparisonRow,
} from './types';
import { getVisibleColumnDefs, type TracesTableMeta } from './columns';
import { getContentColumnMaxSizes } from './getColumnMaxSizes';
import { TraceColumnHeader } from './TraceColumnHeader';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

// No-op sort handlers for non-sortable columns (their header menu omits the sort items anyway).
const noop = () => {};

const getTraceColumnHeaderVariant = (columnDef: TraceTableColumn): TraceTableColumn['headerVariant'] =>
  columnDef.headerVariant === 'standard' || columnDef.headerVariant === 'rich' ? columnDef.headerVariant : undefined;

// Module-local static analytics-id namespace (the `@databricks/no-dynamic-property-value` rule
// requires `componentId` values to be static, so a runtime-injected prefix isn't possible).
const COMPONENT_ID = 'web-shared.traces-table';

// Input/Output are the fill columns: each takes grow factor 1 so any leftover container width is
// split equally between them (no dead whitespace on the right), with `maxWidth: unset` so the cap
// can't block that growth. Every other column stays fixed-width — it occupies exactly its (possibly
// resized) pixel width and horizontal scroll on the Table handles overflow. A drag still sets
// `columnSizing[id]`, which becomes the new flex-basis.
const GROWING_COLUMNS = new Set<string>(['input', 'output']);
/** Grouped mode always shows these and pins them left, in this order. */
export const GROUPED_LEADING_COLUMNS: readonly TraceColumnId[] = ['session', 'input', 'output'];
const columnSizeVariable = (index: number) => `--traces-table-column-${index}`;
const sizeStyleFor = (id: string, index: number): CSSProperties => {
  const width = `var(${columnSizeVariable(index)})`;
  return GROWING_COLUMNS.has(id)
    ? { flex: `1 0 ${width}`, maxWidth: 'unset' }
    : { flex: `0 0 ${width}`, maxWidth: width };
};
const ROW_WIDTH_VARIABLE = '--traces-table-row-width';
const rowWidthStyle = { width: `var(${ROW_WIDTH_VARIABLE})`, minWidth: '100%' } as const;

// Row click opens the drawer; the checkbox cell must not also trigger it.
const stopPropagationProps = {
  onClick: (event: React.MouseEvent) => event.stopPropagation(),
};
// Whether a checkbox change originated from a shift-modified click (range selection). The change
// event's `nativeEvent` carries `shiftKey`; guard structurally since the handler receives `unknown`.
const isShiftModifiedEvent = (event: unknown): boolean => {
  if (typeof event !== 'object' || event === null) {
    return false;
  }
  const nativeEvent = 'nativeEvent' in event ? event.nativeEvent : event;
  return (
    typeof nativeEvent === 'object' &&
    nativeEvent !== null &&
    'shiftKey' in nativeEvent &&
    nativeEvent.shiftKey === true
  );
};
const dataSelectCellAlign = {
  '.table-row-select-cell': { alignItems: 'flex-start' },
  '.table-row-select-cell > *': { transform: 'translateY(2px)' },
} as const;
const headerSelectCellAlign = { '.table-row-select-cell': { alignItems: 'center' } } as const;

// Fixed size of the floating header drag preview (the ghost that follows the cursor while dragging).
const HEADER_DRAG_OVERLAY_WIDTH = 144;
const HEADER_DRAG_OVERLAY_HEIGHT = 36;

// The comparison Run column joins TanStack's sizing state so it resizes and persists like any other
// column; it has no header menu.
const COMPARISON_RUN_COLUMN_ID = 'comparison_run';
const COMPARISON_RUN_COLUMN: TraceTableColumn = {
  id: COMPARISON_RUN_COLUMN_ID,
  size: 140,
  minSize: 100,
  maxSize: 480,
  header: () => null,
  cell: () => null,
};
// In session comparisons the Session column also holds the group hierarchy and each run's identity.
const COMPARISON_SESSION_HIERARCHY_WIDTH = 180;
// A trace can appear in more than one run of a comparison, so a table row is a (trace, run) slot.
const comparisonSlotId = (traceId: string, runUuid: string) => JSON.stringify([traceId, runUuid]);
// Clicks on these inside a matched comparison row keep their own behaviour instead of opening the comparison.
const INTERACTIVE_ELEMENT_SELECTOR =
  'a, button, input, select, textarea, label, [role="button"], [role="link"], [role="checkbox"]';

// The header row is horizontal, so collision is decided purely by pointer-X distance to each header's
// horizontal center — closer center wins. Falls back to `closestCenter` before the pointer is known.
const pointerXAxisCollisionDetection: CollisionDetection = (args) => {
  const pointerX = args.pointerCoordinates?.x;
  if (pointerX === undefined) {
    return closestCenter(args);
  }

  return args.droppableContainers
    .flatMap((droppableContainer) => {
      const rect = args.droppableRects.get(droppableContainer.id);
      if (!rect) {
        return [];
      }
      return [
        {
          id: droppableContainer.id,
          data: {
            droppableContainer,
            value: Math.abs(pointerX - (rect.left + rect.width / 2)),
          },
        },
      ];
    })
    .sort((left, right) => left.data.value - right.data.value);
};

// Lift the floating overlay just above the cursor so it doesn't sit under the pointer.
const positionHeaderDragOverlayAboveCursor: Modifier = ({ draggingNodeRect, transform }) => ({
  ...transform,
  y: transform.y - (draggingNodeRect?.height ?? 0) / 2 - 8,
});

const HEADER_DRAG_OVERLAY_MODIFIERS: Modifiers = [snapCenterToCursor, positionHeaderDragOverlayAboveCursor];

type SortableTraceTableHeaderProps = React.ComponentPropsWithoutRef<typeof TableHeader> & {
  columnId: string;
};

// A `TableHeader` made draggable for column reordering. The activator wraps only the header content
// (not the DuBois resize handle, a sibling), so grabbing the resize handle never starts a reorder; a
// plain click still falls through to the header's sort affordance (the DndContext's PointerSensor only
// activates a drag after 8px of movement).
const SortableTraceTableHeader = ({
  columnId,
  children,
  style,
  ...props
}: SortableTraceTableHeaderProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const [isPressed, setIsPressed] = useState(false);
  const { setNodeRef, setActivatorNodeRef, listeners, transform, transition, isDragging } = useSortable({
    id: columnId,
  });

  useEffect(() => {
    if (!isPressed) {
      return;
    }

    const clearPressed = () => setIsPressed(false);
    window.addEventListener('pointerup', clearPressed);
    window.addEventListener('pointercancel', clearPressed);
    return () => {
      window.removeEventListener('pointerup', clearPressed);
      window.removeEventListener('pointercancel', clearPressed);
    };
  }, [isPressed]);

  const isActive = isPressed || isDragging;

  return (
    <TableHeader
      {...props}
      ref={setNodeRef}
      style={{
        ...style,
        // Translate only — scaling a header to the hovered column's width would also distort its text.
        transform: CSS.Translate.toString(transform),
        transition,
        backgroundColor: isActive ? theme.colors.actionTertiaryBackgroundHover : undefined,
        position: 'relative',
        zIndex: isDragging ? 1 : undefined,
      }}
    >
      <div
        ref={setActivatorNodeRef}
        onPointerDown={(event) => {
          setIsPressed(true);
          listeners?.['onPointerDown']?.(event);
        }}
        onPointerUp={() => setIsPressed(false)}
        onPointerCancel={() => setIsPressed(false)}
        css={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          minWidth: 0,
          cursor: isActive ? 'grabbing' : 'grab',
          touchAction: 'none',
          userSelect: 'none',
        }}
      >
        {children}
      </div>
    </TableHeader>
  );
};

interface TraceTableHeaderDragOverlayProps {
  label: React.ReactNode;
}

// The floating preview shown while dragging a header (rendered into a `DragOverlay` portal).
const TraceTableHeaderDragOverlay = ({ label }: TraceTableHeaderDragOverlayProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      aria-hidden
      css={{
        boxSizing: 'border-box',
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        width: '100%',
        height: '100%',
        padding: `0 ${theme.spacing.sm}px`,
        overflow: 'hidden',
        color: theme.colors.textSecondary,
        backgroundColor: theme.colors.backgroundPrimary,
        border: `2px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
        boxShadow: theme.shadows.sm,
        pointerEvents: 'none',
      }}
    >
      <DragIcon css={{ flexShrink: 0 }} />
      <Typography.Text bold size="sm" ellipsis>
        {label}
      </Typography.Text>
    </div>
  );
};

export interface TracesTableProps {
  traces: ModelTraceInfoV3[];
  /** Consumer-matched input rows, one sub-row per run. Bulk selection is unavailable in this presentation. */
  comparisonRows?: TracesComparisonRow[];
  /** With trace `comparisonRows`: opens a matched row with at least two traces as a comparison. */
  onComparisonRowSelected?: (row: TraceComparisonRow) => void;
  /** With `comparisonRows` grouped by session: the open groups and, within each, the runs whose turns are open. */
  expandedComparisonGroups?: ReadonlySet<string>;
  onToggleComparisonGroup?: (comparisonId: string) => void;
  expandedComparisonRuns?: ReadonlyMap<string, ReadonlySet<string>>;
  onToggleComparisonRun?: (comparisonId: string, runUuid: string) => void;
  visibleColumns: TraceColumnId[];
  /** Product-specific columns appended after the standard columns. Memoize a stable reference. */
  extraColumns?: TraceTableColumn[];
  /** Complete display order, including hidden and product-specific columns. When set, visible columns
   * render in this order; omit it to keep the canonical `visibleColumns` order. */
  columnOrder?: string[];
  /** Persisted per-column pixel widths read once to seed the uncontrolled sizing state (empty map →
   * each column's seeded default). */
  initialColumnSizing: ColumnSizingState;
  /** Persist the final widths on the falling edge of a drag (one write per drag, none on mount). */
  onColumnSizingSettled: (sizing: ColumnSizingState) => void;
  /** True only on first load (no prior data). Background fetches keep prior rows via keepPreviousData. */
  isLoading: boolean;
  /** True while fetching (initial OR transition) — drives aria-busy so SR announces the refresh. */
  isFetching: boolean;
  /** Rows rendered by the skeleton on first load — equals the page size so the swap can't shift height. */
  skeletonRowCount: number;
  onTraceSelected: (trace: ModelTraceInfoV3) => void;
  selectedTraceId?: string;
  /** Cross-page bulk selection keyed by `trace_id`; the table only reads `.has(trace_id)` per row. */
  selectedForBulk: ReadonlyMap<string, ModelTraceInfoV3>;
  isAllOnPageSelected: boolean;
  isSomeOnPageSelected: boolean;
  onToggleBulkRow: (trace: ModelTraceInfoV3, selectRange?: boolean) => void;
  /** Toggle-select every trace in a session header row; omit to disable session-level selection. */
  onToggleBulkRows?: (traces: ModelTraceInfoV3[]) => void;
  onToggleBulkAll: () => void;
  sort: TraceSortColumnId;
  dir: SortDirection;
  /** In comparison rows, any comparison-sortable id, including assessment columns: the consumer sorts the rows. */
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  /** Controls comparison sort affordances only; the consumer sorts the data. */
  enableSorting?: boolean;
  /** Consumer-owned explanation shown on the active comparison header sort toggle. */
  sortTooltip?: string;
  /** Consumer-computed summaries rendered under comparison column titles, keyed by column id. */
  columnHeaderSummaries?: Readonly<Partial<Record<string, TraceColumnHeaderSummary>>>;
  /** Resolves trace-cell links; when absent trace cells open through onTraceSelected. */
  getTraceHref?: TraceHrefGetter;
  /** Resolves the session cell's link destination; when absent the session renders as plain text. */
  getSessionHref?: SessionHrefGetter;
  /** Handles clicks on a grouped session summary row. */
  onSessionSelected?: SessionSelectionHandler;
  /** Toggle a tag filter — wired to the tag pills in the Tags cell; absent → non-clickable pills. */
  onFilterByTag?: (key: string, value: string) => void;
  /** Product-owned renderer for resolving an experiment-scoped run name. */
  renderRunName?: (trace: ModelTraceInfoV3) => React.ReactNode;
  /** Hides the column with the given id — wired to the per-header menu's "Hide column" item. */
  onHideColumn: (columnId: string) => void;
  /** Per-column extra header-menu actions (e.g. "Expand tag columns"), keyed by column id. */
  columnHeaderActions?: Readonly<Partial<Record<string, TraceColumnHeaderAction>>>;
  /** Groups traces with a session id into collapsible session rows. Standalone traces remain rows. */
  isGroupedBySession?: boolean;
  /** Maximum lines shown by input and output previews before truncation. Defaults to one line. */
  previewLineClamp?: number;
  /** Control rendered over the bottom edge of the header row, such as a toggle for rich headers. */
  headerBottomOverlay?: React.ReactNode;
  /** Reorders columns when a user drags one header onto another; absent → headers aren't draggable. */
  onReorderColumn?: (activeColumn: string, targetColumn: string) => void;
  /** When true, session-level column aggregates (e.g. token totals) are suppressed because the
   * visible page may not contain all traces for every session. Set when isGroupedBySession is true
   * and there are multiple pages of results (hasNext || hasPrev). Mirrors the onToggleBulkRows
   * suppression pattern: rather than show a silently-partial number, we show nothing. */
  sessionsMayBeIncomplete?: boolean;
}

interface GroupedTraceRows {
  sessionId?: string;
  rows: Row<ModelTraceInfoV3>[];
}

// The V4 long identifier is the selection key: the consumer stores the same id when a row is opened,
// so comparing against it highlights the correct row.
const rowIdFor = (trace: ModelTraceInfoV3) =>
  doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;

/**
 * Presentational, fully-controlled TanStack-backed `Table` for traces: fixed-width, user-resizable
 * columns (widths persisted by the consumer), single-line truncation, row selection persisted across
 * pages, and sort affordances on the only two server-sortable columns (start time, duration). The
 * leading select column is a fixed cell outside the resizable column set, so header, data, and
 * skeleton rows share an identical leading cell and stay column-aligned. First load renders exactly
 * `skeletonRowCount` skeleton rows under the real header so swapping in real rows causes no layout shift.
 *
 * `React.memo`'d: its props are stable across a search keystroke (search text goes only to the
 * toolbar), so a keystroke doesn't re-render the table; a bulk-select click changes `selectedForBulk`'s
 * identity and re-renders it (to repaint checkboxes), but the memoized cells skip re-rendering. `memo`
 * only gates parent-triggered re-renders — live column resize is driven by the table instance's
 * internal state inside this component and is unaffected.
 */
export const TracesTable: React.MemoExoticComponent<(props: TracesTableProps) => JSX.Element> = memo(
  function TracesTable({
    traces,
    comparisonRows,
    onComparisonRowSelected,
    expandedComparisonGroups,
    onToggleComparisonGroup,
    expandedComparisonRuns,
    onToggleComparisonRun,
    visibleColumns,
    extraColumns,
    columnOrder,
    initialColumnSizing,
    onColumnSizingSettled,
    isLoading,
    isFetching,
    skeletonRowCount,
    onTraceSelected,
    selectedTraceId,
    selectedForBulk,
    isAllOnPageSelected,
    isSomeOnPageSelected,
    onToggleBulkRow,
    onToggleBulkRows,
    onToggleBulkAll,
    sort,
    dir,
    onSort,
    enableSorting = !comparisonRows,
    sortTooltip,
    columnHeaderSummaries,
    getTraceHref,
    getSessionHref,
    onSessionSelected,
    onFilterByTag,
    renderRunName,
    onHideColumn,
    columnHeaderActions,
    isGroupedBySession = false,
    previewLineClamp = 1,
    headerBottomOverlay,
    onReorderColumn,
    sessionsMayBeIncomplete = false,
  }: TracesTableProps) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const isComparison = comparisonRows !== undefined;
    const isSessionComparison = isComparison && isGroupedBySession;
    // Keep the checkbox centered and leave a full spacing token before the first data column.
    const selectCellCss = {
      '.table-row-select-cell': { alignItems: 'center', paddingRight: theme.spacing.sm },
    } as const;

    // Canonical-order visible column defs + any product columns. `extraColumns` is guarded to a stable
    // reference so a stable/undefined value doesn't defeat the deep memo (see `getVisibleColumnDefs`).
    // When grouped, session + input/output are pinned left. When column reordering is enabled, columnOrder
    // overrides the canonical order.
    const columns = useMemo(() => {
      // Grouped mode always shows session + the input/output previews and pins them left, so a session
      // header reads left-to-right as "which session → first input → last output" regardless of the
      // user's own column visibility/order.
      const groupedVisibleColumns = isGroupedBySession
        ? [...new Set([...visibleColumns, ...GROUPED_LEADING_COLUMNS])]
        : visibleColumns;
      const visibleColumnDefs = getVisibleColumnDefs(groupedVisibleColumns, extraColumns, columnOrder);
      const groupedColumnRank = new Map<string, number>(GROUPED_LEADING_COLUMNS.map((id, index) => [id, index]));
      const getGroupedColumnRank = (id: string | undefined) =>
        id === undefined ? Infinity : (groupedColumnRank.get(id) ?? Infinity);
      const orderedVisibleColumnDefs = isGroupedBySession
        ? [...visibleColumnDefs].sort((left, right) => getGroupedColumnRank(left.id) - getGroupedColumnRank(right.id))
        : visibleColumnDefs;
      // Comparison rows pin Input first, then the Run identity column, then the value columns.
      const presentedColumnDefs =
        isComparison && !isSessionComparison
          ? [
              ...orderedVisibleColumnDefs.filter((column) => column.id === 'input'),
              COMPARISON_RUN_COLUMN,
              ...orderedVisibleColumnDefs.filter((column) => column.id !== 'input'),
            ]
          : orderedVisibleColumnDefs;
      // Product-specific column ids are intentionally absent and resolve to undefined.
      const contentMaxSizes: Readonly<Record<string, number | undefined>> = getContentColumnMaxSizes(traces, intl);
      return presentedColumnDefs.map((column) => {
        if (column.id === undefined) {
          return column;
        }
        const contentMaxSize = contentMaxSizes[column.id];
        const persistedSize = initialColumnSizing[column.id];
        if (isSessionComparison && column.id === 'session') {
          // The hierarchy and run identity share the resizable Session column in comparison mode.
          return {
            ...column,
            minSize: Math.max(column.minSize ?? 0, COMPARISON_SESSION_HIERARCHY_WIDTH),
            size: (column.size ?? 0) + COMPARISON_SESSION_HIERARCHY_WIDTH,
            maxSize: Math.max(
              column.maxSize ?? 0,
              (contentMaxSize ?? 0) + COMPARISON_SESSION_HIERARCHY_WIDTH,
              persistedSize ?? 0,
            ),
          };
        }
        const summaryMinWidth = columnHeaderSummaries?.[column.id]?.minWidth;
        if (summaryMinWidth !== undefined) {
          // A header summary must fit every run's value, so it sets the column's minimum width.
          const minSize = Math.max(column.minSize ?? 0, summaryMinWidth);
          const maxSize = contentMaxSize ?? Math.max(column.maxSize ?? 0, persistedSize ?? 0);
          return { ...column, minSize, size: Math.max(column.size ?? 0, minSize), maxSize: Math.max(maxSize, minSize) };
        }
        if (contentMaxSize === undefined && persistedSize === undefined) {
          return column;
        }
        // Match the established StatementsTable layout: compact columns are fixed to their measured
        // content size, while the useful text columns flex into the remaining container width. The
        // content-derived size is therefore the real ceiling, not a lower bound beneath a larger
        // static/persisted max. Product columns have no shared content measurement and retain their
        // own declared/persisted ceiling.
        const maxSize = contentMaxSize ?? Math.max(column.maxSize ?? 0, persistedSize ?? 0);
        return { ...column, maxSize };
      });
    }, [
      visibleColumns,
      extraColumns,
      isGroupedBySession,
      isComparison,
      isSessionComparison,
      traces,
      intl,
      initialColumnSizing,
      columnOrder,
      columnHeaderSummaries,
    ]);

    // Product-owned per-column session summary renderers (e.g. assessment aggregates), keyed by
    // column id so a session header cell can look one up.
    const sessionCellRenderers = useMemo(
      () =>
        new Map(
          columns.flatMap((column) =>
            column.id && column.renderSessionCell ? [[column.id, column.renderSessionCell] as const] : [],
          ),
        ),
      [columns],
    );

    // Matched trace rows with at least two traces open as a comparison; their trace links do the same.
    const comparisonRowByTraceId = useMemo(() => {
      const rowsByTraceId = new Map<string, TraceComparisonRow>();
      for (const comparison of comparisonRows ?? []) {
        if (comparison.kind === 'session' || comparison.runs.filter((run) => run.trace).length < 2) {
          continue;
        }
        for (const run of comparison.runs) {
          if (run.trace) {
            rowsByTraceId.set(rowIdFor(run.trace), comparison);
          }
        }
      }
      return rowsByTraceId;
    }, [comparisonRows]);
    const opensComparisonRows = Boolean(onComparisonRowSelected) && comparisonRowByTraceId.size > 0;
    // Without activatable comparison rows the table keeps the caller's own handlers.
    const metaOnTraceSelected = useMemo(
      () =>
        opensComparisonRows
          ? (trace: ModelTraceInfoV3) => {
              const comparison = comparisonRowByTraceId.get(rowIdFor(trace));
              if (comparison && onComparisonRowSelected) {
                onComparisonRowSelected(comparison);
                return;
              }
              onTraceSelected(trace);
            }
          : onTraceSelected,
      [opensComparisonRows, comparisonRowByTraceId, onComparisonRowSelected, onTraceSelected],
    );
    const metaGetTraceHref = useMemo(
      () =>
        opensComparisonRows
          ? (trace: ModelTraceInfoV3) =>
              comparisonRowByTraceId.has(rowIdFor(trace)) ? undefined : getTraceHref?.(trace)
          : getTraceHref,
      [opensComparisonRows, comparisonRowByTraceId, getTraceHref],
    );
    const meta = useMemo<TracesTableMeta>(
      () => ({
        intl,
        onTraceSelected: metaOnTraceSelected,
        getTraceHref: metaGetTraceHref,
        getSessionHref,
        onFilterByTag,
        renderRunName,
        previewLineClamp,
      }),
      [intl, metaOnTraceSelected, metaGetTraceHref, getSessionHref, onFilterByTag, renderRunName, previewLineClamp],
    );

    // Clamp the state TanStack is seeded with as well as the column definition. `getSize()` normally
    // applies maxSize, but retaining an oversized raw initial value can still leak into the flex-table
    // layout before the first resize update and leave large blank gaps between compact columns.
    const normalizedInitialColumnSizing = useMemo(() => {
      const sizing = { ...initialColumnSizing };
      for (const column of columns) {
        if (column.id !== undefined && sizing[column.id] !== undefined && column.maxSize !== undefined) {
          sizing[column.id] = Math.min(sizing[column.id], column.maxSize);
        }
      }
      return sizing;
    }, [columns, initialColumnSizing]);

    const comparisonEntries = useMemo(
      () =>
        comparisonRows?.flatMap((row) =>
          row.runs.flatMap((run) =>
            ('traces' in run ? run.traces : run.trace ? [run.trace] : []).map((trace) => ({
              id: comparisonSlotId(trace.trace_id, run.runUuid),
              trace,
            })),
          ),
        ),
      [comparisonRows],
    );
    const tableTraces = useMemo(
      () => comparisonEntries?.map(({ trace }) => trace) ?? traces,
      [comparisonEntries, traces],
    );
    const getComparisonRowId = useCallback(
      (_trace: ModelTraceInfoV3, index: number) => comparisonEntries?.[index]?.id ?? String(index),
      [comparisonEntries],
    );

    const table = useReactTable_unverifiedWithReact18<ModelTraceInfoV3>('traces-table/TracesTable.tsx', {
      data: tableTraces,
      columns,
      getCoreRowModel: getCoreRowModel(),
      getRowId: comparisonEntries ? getComparisonRowId : rowIdFor,
      enableColumnResizing: true,
      // Keep 'onChange' for live drag feedback (the table's internal state re-renders per tick and
      // `getSize()` tracks the cursor). Sizing is uncontrolled — seeded once via `initialState` — so a
      // drag no longer writes persistence on every mousemove.
      columnResizeMode: 'onChange',
      initialState: { columnSizing: normalizedInitialColumnSizing },
      meta,
    });

    const [expandedSessions, setExpandedSessions] = useState<Set<string>>(() => new Set());
    const toggleSessionExpanded = useCallback((sessionId: string) => {
      setExpandedSessions((current) => {
        const next = new Set(current);
        if (next.has(sessionId)) {
          next.delete(sessionId);
        } else {
          next.add(sessionId);
        }
        return next;
      });
    }, []);

    // In grouped mode, bucket the (already sorted) rows by session id, preserving first-seen session
    // order; standalone traces (no session id) stay their own single-row groups in place. Traces
    // within a session are re-sorted oldest-first so an expanded session reads turn 1 → N top-down.
    const tableRows = table.getRowModel().rows;
    const groupedRows = useMemo(() => {
      if (!isGroupedBySession) {
        return undefined;
      }
      const sessions = new Map<string, Row<ModelTraceInfoV3>[]>();
      const orderedGroups: GroupedTraceRows[] = [];
      for (const row of tableRows) {
        const sessionId = row.original.trace_metadata?.[SESSION_ID_METADATA_KEY];
        if (!sessionId) {
          orderedGroups.push({ rows: [row] });
          continue;
        }
        let sessionRows = sessions.get(sessionId);
        if (!sessionRows) {
          sessionRows = [];
          sessions.set(sessionId, sessionRows);
          orderedGroups.push({ sessionId, rows: sessionRows });
        }
        sessionRows.push(row);
      }
      sessions.forEach((rows) => {
        rows.sort((left, right) => {
          const leftStartTime = Date.parse(left.original.request_time);
          const rightStartTime = Date.parse(right.original.request_time);
          if (Number.isNaN(leftStartTime)) {
            return Number.isNaN(rightStartTime) ? 0 : 1;
          }
          return Number.isNaN(rightStartTime) ? -1 : leftStartTime - rightStartTime;
        });
      });
      return orderedGroups;
    }, [isGroupedBySession, tableRows]);

    // Persist only on the falling edge of a resize (mouseup): one write per drag, none on mount. The
    // falling-edge ref avoids the redundant mount write that a plain `if (!isResizing) persist(...)`
    // would do.
    const isResizingColumn = Boolean(table.getState().columnSizingInfo.isResizingColumn);
    const wasResizing = useRef(false);
    useEffect(() => {
      if (wasResizing.current && !isResizingColumn) {
        // TanStack stores the raw pointer delta in `columnSizing`, even when `getSize()` clamps the
        // rendered width to the column's min/max. Reset that raw state before persisting it; otherwise
        // an oversized first drag becomes the next render's persisted max and a second drag can grow
        // beyond the original ceiling.
        const columnSizing = table.getState().columnSizing;
        const clampedSizing = { ...columnSizing };
        for (const column of table.getAllLeafColumns()) {
          if (columnSizing[column.id] !== undefined) {
            clampedSizing[column.id] = column.getSize();
          }
        }
        table.setColumnSizing(clampedSizing);
        onColumnSizingSettled(clampedSizing);
      }
      wasResizing.current = isResizingColumn;
    }, [table, isResizingColumn, onColumnSizingSettled]);

    const leafHeaders = table.getLeafHeaders();

    // A density floor matching the preview clamp keeps Standard/Tall rows uniform and lets the loading
    // skeleton reserve the height real rows will take (no layout shift on swap). Gate it on a visible
    // preview column: with both hidden, the floor would force tall, empty rows, so those size to content.
    const isPreviewColumnVisible = leafHeaders.some(
      (header) => header.column.id === 'input' || header.column.id === 'output',
    );
    const previewLineHeight = Number.parseInt(theme.typography.lineHeightBase, 10);
    const rowMinHeight =
      previewLineClamp > 1 && isPreviewColumnVisible
        ? theme.general.heightSm + theme.spacing.md + (previewLineClamp - 2) * previewLineHeight
        : undefined;
    const baseDataRowStyle: CSSProperties = rowMinHeight
      ? { ...rowWidthStyle, minHeight: rowMinHeight }
      : rowWidthStyle;
    // Each row checkbox cell carries the design system's inline z-index (zIndexBase), which paints it over
    // the sticky header once the row scrolls under it, so a click on select-all lands on that row's
    // checkbox. Isolating each body row keeps that z-index inside the row.
    const isolateBodyRows = shouldEnableNewEvalRunsUi();
    const dataRowStyle: CSSProperties = isolateBodyRows
      ? { ...baseDataRowStyle, isolation: 'isolate' }
      : baseDataRowStyle;
    const sessionRowStyle: CSSProperties = isolateBodyRows ? { ...rowWidthStyle, isolation: 'isolate' } : rowWidthStyle;

    // 8px activation distance: a pointerdown/up without moving 8px is a click (header sort fires); only
    // a real drag starts a reorder. This is the primary guard against sort-clicks becoming drags.
    const [activeDragColumnId, setActiveDragColumnId] = useState<string | null>(null);
    const headerDragSensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 8 } }));
    const handleHeaderDragStart = useCallback(({ active }: DragStartEvent) => {
      setActiveDragColumnId(typeof active.id === 'string' ? active.id : null);
    }, []);
    const handleHeaderDragEnd = useCallback(
      ({ active, over }: DragEndEvent) => {
        setActiveDragColumnId(null);
        if (
          onReorderColumn &&
          over !== null &&
          active.id !== over.id &&
          typeof active.id === 'string' &&
          typeof over.id === 'string'
        ) {
          onReorderColumn(active.id, over.id);
        }
      },
      [onReorderColumn],
    );
    const activeDragHeader = activeDragColumnId
      ? leafHeaders.find(({ column }) => column.id === activeDragColumnId)
      : undefined;
    const activeDragHeaderLabel = activeDragHeader
      ? flexRender(activeDragHeader.column.columnDef.header, activeDragHeader.getContext())
      : null;
    const reorderableHeaderIds = leafHeaders.map(({ column }) => column.id);

    // Pin every row to the summed width of the visible columns so its hover/selected background spans
    // the full horizontal extent, not just the visible viewport. A DuBois `scrollable` Table makes the
    // table its own horizontal scroll container, and each `TableRow` is a `display: flex` block with no
    // explicit width — so it's only as wide as the scroll *viewport*. The cells are `flex-shrink: 0`, so
    // once the columns overflow they extend past the row box and the row background (which paints only to
    // the row's own width) cuts off at the viewport's right edge. `minWidth: 100%` keeps rows filling the
    // viewport — and lets the input/output fill columns grow — when the columns are narrower than it.
    //
    // The leading select cell lives outside the TanStack column set: 16px checkbox content plus 8px
    // padding on each side, including the explicit trailing gap before the first data column.
    // Comparison rows have no select cell: Run participates in the column model instead.
    const selectCellWidth = isComparison ? 0 : theme.spacing.md + theme.spacing.sm * 2;
    // Grouped mode inserts a leading expand/collapse toggle cell (a small button) before the columns;
    // reserve its width so header, session, and trace rows all stay column-aligned.
    const sessionToggleWidth = isGroupedBySession && !isComparison ? theme.general.heightSm + theme.spacing.xs : 0;
    const rowWidth = leafHeaders.reduce(
      (total, header) => total + header.getSize(),
      selectCellWidth + sessionToggleWidth,
    );
    // Publish live widths once on the scroll container. Header/cell styles reference these variables,
    // so React updates one DOM node per resize tick instead of diffing a new inline style on every cell.
    // This follows ProcessListTable's established live-resize performance pattern.
    const tableSizeVariables = {
      [ROW_WIDTH_VARIABLE]: `${rowWidth}px`,
      ...Object.fromEntries(leafHeaders.map((header, index) => [columnSizeVariable(index), `${header.getSize()}px`])),
    } as CSSProperties;
    const rowPaddingCss: CSSObject = {
      '&& > *': {
        paddingTop: theme.spacing.xs + 2,
        paddingBottom: theme.spacing.xs + 2,
      },
    };
    const columnStyles = useMemo(
      () => new Map(leafHeaders.map((header, index) => [header.column.id, sizeStyleFor(header.column.id, index)])),
      // Header identities are stable while only column sizing changes.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [table, columns],
    );

    const renderHeaderBottomOverlay = () => (
      <div
        data-testid="traces-table-header-bottom-overlay"
        css={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: -theme.spacing.xs,
          padding: 0,
          // The row gives each child a bottom border; this one hangs below the header, so its border
          // would draw a second line under the header's own.
          borderBottom: 'none',
          zIndex: 101,
          pointerEvents: 'none',
        }}
      >
        {headerBottomOverlay}
      </div>
    );
    // Leaves room under the header for the part of the overlay that hangs below it.
    const renderHeaderBottomOverlaySpacer = () => <div aria-hidden css={{ height: theme.spacing.sm, flexShrink: 0 }} />;

    // Above the isolated body rows and the sticky session rows (see `isolateBodyRows`).
    const headerRowStyle = isolateBodyRows ? { ...rowWidthStyle, zIndex: 2 } : rowWidthStyle;

    const hasRichHeaders = leafHeaders.some(
      (header) => getTraceColumnHeaderVariant(header.column.columnDef as TraceTableColumn) === 'rich',
    );

    // Header-row overrides: align its contents and set the header's bottom border.
    const headerRowCss = {
      ...selectCellCss,
      ...(hasRichHeaders ? { '.table-row-select-cell > *': { transform: 'translateY(2px)' } } : {}),
      borderBottom: `1px solid ${theme.colors.border}`,
      // Center each header label against the (vertically-centered) select-all checkbox. Padding sets
      // the label→divider gap directly — DS applies --table-row-vertical-padding via an inline style,
      // so a `css` override of it is a no-op. Rich headers make a tall row, so labels align to its top.
      '[role="columnheader"]': {
        alignItems: hasRichHeaders ? 'flex-start' : 'center',
        paddingBottom: theme.spacing.mid - 2,
        '&:hover .traces-table-header-menu-trigger, &:focus-within .traces-table-header-menu-trigger': {
          opacity: 1,
        },
      },
      // TableRowSelectCell resets vertical padding to zero, so restore the same padding as the other
      // header cells to keep the checkbox and labels on the same center line.
      '&& .table-row-select-cell': {
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.mid - 2,
      },
      // Keep the header's select-all checkbox always visible (data rows stay hover-reveal).
      '&& .table-row-select-cell input[type="checkbox"] ~ *': { opacity: 1 },
    };

    // Empty cell matching the leading session-toggle button's width, so rows without a toggle (header,
    // skeleton, expanded trace rows) keep their columns aligned under the session rows that do.
    const renderSessionToggleSpacer = () => <div css={{ width: sessionToggleWidth, flexShrink: 0 }} />;

    const renderSessionPreview = (value: string, color: 'primary' | 'secondary' = 'primary') =>
      value ? (
        <Typography.Text color={color} ellipsis>
          {value}
        </Typography.Text>
      ) : (
        <Typography.Text color="secondary">-</Typography.Text>
      );

    const renderSessionHeaderCell = (sessionId: string, trace: ModelTraceInfoV3) => {
      const tag = (
        <Tag componentId={`${COMPONENT_ID}.session-id`} title={sessionId} css={{ maxWidth: '100%' }}>
          <SpeechBubbleIcon css={{ fontSize: theme.typography.fontSizeBase, marginRight: theme.spacing.xs }} />
          <Typography.Text ellipsis>{sessionId}</Typography.Text>
        </Tag>
      );
      const sessionHref = getSessionHref?.({ trace, sessionId });
      return sessionHref ? (
        <Link
          componentId={`${COMPONENT_ID}.session-link`}
          to={sessionHref}
          onClick={(event) => event.stopPropagation()}
        >
          {tag}
        </Link>
      ) : (
        tag
      );
    };

    const renderTraceRow = (
      row: Row<ModelTraceInfoV3>,
      includeSessionToggleSpacer = false,
      sessionTurnNumber?: number,
    ) => {
      const isSelected = row.id === selectedTraceId;
      const isBulkChecked = selectedForBulk.has(row.original.trace_id);
      return (
        <TableRow
          key={row.id}
          onClick={(event) => {
            const traceHref = getTraceHref?.(row.original);
            if ((event.ctrlKey || event.metaKey) && traceHref) {
              event.preventDefault();
              window.open(
                typeof traceHref === 'string' ? traceHref : createPath(traceHref),
                '_blank',
                'noopener,noreferrer',
              );
              return;
            }
            onTraceSelected(row.original);
          }}
          style={dataRowStyle}
          css={{
            cursor: 'pointer',
            backgroundColor: isSelected ? theme.colors.tableBackgroundUnselectedHover : undefined,
            ...rowPaddingCss,
            ...dataSelectCellAlign,
          }}
        >
          <TableRowSelectCell
            componentId={`${COMPONENT_ID}.row-select`}
            checked={isBulkChecked}
            onChange={(event) => onToggleBulkRow(row.original, !isGroupedBySession && isShiftModifiedEvent(event))}
            checkboxLabel={intl.formatMessage(
              {
                defaultMessage: 'Select trace {traceId}',
                description: 'Aria label for the per-row select checkbox in the traces table',
              },
              { traceId: row.original.trace_id },
            )}
            {...stopPropagationProps}
          />
          {includeSessionToggleSpacer && renderSessionToggleSpacer()}
          {row.getVisibleCells().map((cell) => (
            <TableCell key={cell.id} css={{ verticalAlign: 'middle' }} style={columnStyles.get(cell.column.id)}>
              {cell.column.id === 'session' && sessionTurnNumber !== undefined ? (
                <Tag componentId={`${COMPONENT_ID}.session-turn`}>
                  {intl.formatMessage(
                    {
                      defaultMessage: 'Turn {turnNumber}',
                      description: 'Sequential turn number for a trace within an expanded session',
                    },
                    { turnNumber: sessionTurnNumber },
                  )}
                </Tag>
              ) : (
                flexRender(cell.column.columnDef.cell, cell.getContext())
              )}
            </TableCell>
          ))}
        </TableRow>
      );
    };

    // Comparison rows: one grid per matched input owns every run's cells, so wrapped values can't
    // misalign sibling columns. The Input cell spans the run sub-rows and stays pinned while scrolling.
    const pinnedComparisonInputStyle = {
      position: 'sticky',
      left: 0,
      zIndex: 1,
      backgroundColor: theme.colors.backgroundPrimary,
    } as const;
    const comparisonInputHeader =
      isComparison && !isSessionComparison ? leafHeaders.find((header) => header.column.id === 'input') : undefined;
    const comparisonRunHeader = isComparison
      ? leafHeaders.find((header) => header.column.id === COMPARISON_RUN_COLUMN_ID)
      : undefined;
    const comparisonValueHeaders = leafHeaders.filter(
      (header) =>
        (isSessionComparison || header.column.id !== 'input') && header.column.id !== COMPARISON_RUN_COLUMN_ID,
    );
    const hasSummaryHeaders = leafHeaders.some((header) => Boolean(columnHeaderSummaries?.[header.column.id]));
    const growingComparisonColumnCount = leafHeaders.filter((header) => GROWING_COLUMNS.has(header.column.id)).length;
    const getComparisonColumnWidth = (header: (typeof leafHeaders)[number]) => {
      const width = `var(${columnSizeVariable(leafHeaders.indexOf(header))})`;
      // Match the headers' flex-basis plus an equal share of spare width, even after an unequal resize.
      return GROWING_COLUMNS.has(header.column.id)
        ? `calc(${width} + (100% - var(${ROW_WIDTH_VARIABLE})) / ${growingComparisonColumnCount})`
        : width;
    };
    const comparisonGridColumns = [
      ...(comparisonInputHeader ? [getComparisonColumnWidth(comparisonInputHeader)] : []),
      ...(comparisonRunHeader ? [getComparisonColumnWidth(comparisonRunHeader)] : []),
      ...comparisonValueHeaders.map(getComparisonColumnWidth),
    ].join(' ');
    const comparisonRunGridColumn = comparisonInputHeader ? 2 : 1;
    const comparisonValueGridColumnOffset = comparisonRunHeader ? comparisonRunGridColumn + 1 : 1;
    const comparisonCellRenderers = new Map(
      columns.flatMap((column) =>
        column.id && column.renderComparisonCell ? [[column.id, column.renderComparisonCell] as const] : [],
      ),
    );
    const comparisonCellCss = { alignContent: 'center', alignItems: 'center' } as const;
    const comparisonHeaderRowCss = {
      ...headerRowCss,
      '[role="columnheader"]': {
        ...headerRowCss['[role="columnheader"]'],
        alignItems: hasRichHeaders || hasSummaryHeaders ? 'flex-start' : 'center',
      },
      ...(hasSummaryHeaders
        ? {
            '[role="columnheader"] > [role="button"][aria-label="Resize Column"]': {
              // Extend the drag target past its fixed height to match the summary separators.
              minHeight: `calc(100% - ${2 * theme.spacing.sm}px)`,
            },
          }
        : {}),
    };

    const renderComparisonHeader = (header: (typeof leafHeaders)[number]) => {
      const columnId = header.column.id;
      const columnDef = header.column.columnDef as TraceTableColumn;
      const labelNode = flexRender(columnDef.header, header.getContext());
      const isSortable = enableSorting && isComparisonSortableTraceColumn(columnId);
      // The consumer sorts comparison rows itself, so it accepts every comparison-sortable id.
      const sortComparison = onSort as (column: TraceSortColumnId, direction: SortDirection) => void;
      return (
        <TableHeader
          key={header.id}
          componentId={`${COMPONENT_ID}.header`}
          header={header}
          column={header.column}
          setColumnSizing={table.setColumnSizing}
          style={{
            ...columnStyles.get(columnId),
            ...(columnId === 'input' && !isSessionComparison
              ? { ...pinnedComparisonInputStyle, zIndex: 2 }
              : undefined),
          }}
          wrapContent={false}
        >
          {hasSummaryHeaders && (
            <div
              aria-hidden
              data-testid="traces-table-comparison-summary-border"
              css={{
                position: 'absolute',
                insetBlock: theme.spacing.sm,
                insetInline: 0,
                borderInlineStart: `1px solid ${theme.colors.borderDecorative}`,
                // Adjacent headers share one separator instead of doubling its width.
                borderInlineEnd:
                  columnId === 'input' || leafHeaders.indexOf(header) === leafHeaders.length - 1
                    ? `1px solid ${theme.colors.borderDecorative}`
                    : undefined,
                pointerEvents: 'none',
              }}
            />
          )}
          <TraceColumnHeader
            columnId={columnId}
            label={labelNode}
            labelText={columnDef.labelText ?? (typeof labelNode === 'string' ? labelNode : undefined)}
            sortable={isSortable}
            sortDirection={isSortable && sort === columnId ? dir : 'none'}
            sortTooltip={sortTooltip}
            onSortAscending={isSortable ? () => sortComparison(columnId, 'asc') : noop}
            onSortDescending={isSortable ? () => sortComparison(columnId, 'desc') : noop}
            // Session comparisons always show the grouped leading columns, so they can't be hidden.
            onHide={
              isSessionComparison && GROUPED_LEADING_COLUMNS.includes(columnId as TraceColumnId)
                ? undefined
                : () => onHideColumn(columnId)
            }
            action={columnHeaderActions?.[columnId]}
            menuContent={columnDef.headerMenuContent}
            headerVariant={getTraceColumnHeaderVariant(columnDef)}
            summary={columnHeaderSummaries?.[columnId]?.content}
            titleMinHeight={hasSummaryHeaders ? theme.general.heightSm : undefined}
          />
        </TableHeader>
      );
    };

    const renderComparisonRunLabel = (run: TraceComparisonRow['runs'][number], onOpenComparison?: () => void) => {
      const { trace } = run;
      const traceHref = trace ? getTraceHref?.(trace) : undefined;
      const label = <Typography.Text ellipsis>{run.label}</Typography.Text>;
      return (
        <Tooltip componentId={`${COMPONENT_ID}.comparison-run-tooltip`} content={run.label}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
            <span
              aria-hidden
              css={{
                backgroundColor: run.color,
                width: theme.spacing.sm,
                height: theme.spacing.sm,
                borderRadius: '50%',
                flexShrink: 0,
              }}
            />
            {onOpenComparison ? (
              <Button
                componentId={`${COMPONENT_ID}.open-comparison`}
                type="link"
                onClick={onOpenComparison}
                css={{ width: '100%', minWidth: 0, textAlign: 'left' }}
                dangerouslyAppendWrapperCss={{ width: '100%', minWidth: 0 }}
              >
                {label}
              </Button>
            ) : trace && traceHref ? (
              <Link
                componentId={`${COMPONENT_ID}.comparison-run-trace`}
                to={traceHref}
                css={{ minWidth: 0 }}
                onClick={(event) => {
                  // A plain click opens the trace in place; modified clicks follow the link to a new tab.
                  if (!event.metaKey && !event.ctrlKey && !event.shiftKey && event.button === 0) {
                    event.preventDefault();
                    onTraceSelected(trace);
                  }
                }}
              >
                {label}
              </Link>
            ) : trace ? (
              <Button
                componentId={`${COMPONENT_ID}.comparison-run-trace`}
                type="link"
                onClick={() => onTraceSelected(trace)}
                css={{ width: '100%', minWidth: 0, textAlign: 'left' }}
                dangerouslyAppendWrapperCss={{ width: '100%', minWidth: 0 }}
              >
                {label}
              </Button>
            ) : (
              label
            )}
          </div>
        </Tooltip>
      );
    };

    const renderComparisonRow = (comparison: TraceComparisonRow) => {
      const canOpenComparison = Boolean(
        onComparisonRowSelected && comparison.runs.filter((run) => run.trace).length >= 2,
      );
      const openComparison = canOpenComparison ? () => onComparisonRowSelected?.(comparison) : undefined;
      const openComparisonLabel = intl.formatMessage({
        defaultMessage: 'Open trace comparison',
        description: 'Accessible label for a matched trace comparison row',
      });
      const comparisonInputContent = openComparison ? (
        <Button
          componentId={`${COMPONENT_ID}.open-comparison`}
          type="link"
          aria-label={openComparisonLabel}
          onClick={openComparison}
          css={{ width: '100%', minWidth: 0, textAlign: 'left' }}
          // The Button centers its content span, which won't shrink below its text; let it fill and shrink
          // so the preview starts at the left edge and truncates instead of clipping on both sides.
          dangerouslyAppendWrapperCss={{ width: '100%', minWidth: 0 }}
        >
          {renderSessionPreview(comparison.input)}
        </Button>
      ) : (
        renderSessionPreview(comparison.input)
      );
      return (
        <div
          role="rowgroup"
          aria-label={openComparison ? openComparisonLabel : undefined}
          onClick={
            openComparison
              ? (event) => {
                  const target = event.target;
                  if (target instanceof Element && target.closest(INTERACTIVE_ELEMENT_SELECTOR)) {
                    return;
                  }
                  openComparison();
                }
              : undefined
          }
          key={comparison.id}
          style={{ ...rowWidthStyle, display: 'grid', gridTemplateColumns: comparisonGridColumns }}
          css={{
            borderBottom: `1px solid ${theme.colors.border}`,
            '& > [role="row"] > [role="cell"]': { borderBottom: 0 },
            ...(openComparison && {
              cursor: 'pointer',
              '&:hover > [role="row"] > [role="cell"]': {
                backgroundColor: theme.colors.tableBackgroundUnselectedHover,
              },
            }),
          }}
        >
          {comparison.runs.map((run, runIndex) => {
            const { trace } = run;
            const row = trace ? table.getRow(comparisonSlotId(trace.trace_id, run.runUuid)) : undefined;
            const isSelected = Boolean(trace && rowIdFor(trace) === selectedTraceId);
            return (
              <TableRow key={run.runUuid} style={{ display: 'contents' }} css={rowPaddingCss}>
                {comparisonInputHeader && runIndex === 0 && (
                  <TableCell
                    wrapContent={false}
                    aria-label={comparison.input || undefined}
                    aria-rowspan={comparison.runs.length}
                    style={{
                      gridColumn: 1,
                      gridRow: `1 / span ${comparison.runs.length}`,
                      minWidth: 0,
                      ...pinnedComparisonInputStyle,
                    }}
                    css={comparisonCellCss}
                  >
                    {comparisonInputContent}
                  </TableCell>
                )}
                {comparisonRunHeader && (
                  <TableCell
                    wrapContent={false}
                    style={{
                      gridColumn: comparisonRunGridColumn,
                      gridRow: runIndex + 1,
                      minWidth: 0,
                      minHeight: rowMinHeight,
                    }}
                    css={comparisonCellCss}
                  >
                    {renderComparisonRunLabel(run, openComparison)}
                  </TableCell>
                )}
                {comparisonValueHeaders.map((header, columnIndex) => {
                  // Session comparisons have no pinned Input column, so the input spans the run sub-rows here.
                  const isInput = isSessionComparison && header.column.id === 'input';
                  if (isInput && runIndex > 0) {
                    return null;
                  }
                  const cell = row?.getVisibleCells().find((candidate) => candidate.column.id === header.column.id);
                  const renderComparisonCell = comparisonCellRenderers.get(header.column.id);
                  return (
                    <TableCell
                      key={header.id}
                      wrapContent={false}
                      aria-rowspan={isInput ? comparison.runs.length : undefined}
                      style={{
                        gridColumn: comparisonValueGridColumnOffset + columnIndex,
                        gridRow: isInput ? `1 / span ${comparison.runs.length}` : runIndex + 1,
                        minWidth: 0,
                        backgroundColor:
                          !isInput && isSelected ? theme.colors.tableBackgroundUnselectedHover : undefined,
                      }}
                      css={comparisonCellCss}
                    >
                      {isSessionComparison && header.column.id === 'session' ? (
                        renderComparisonRunLabel(run, openComparison)
                      ) : isInput ? (
                        comparisonInputContent
                      ) : cell && trace ? (
                        renderComparisonCell ? (
                          renderComparisonCell({ trace, row: comparison, runIndex })
                        ) : (
                          flexRender(cell.column.columnDef.cell, cell.getContext())
                        )
                      ) : (
                        <Typography.Text color="secondary">-</Typography.Text>
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
        </div>
      );
    };

    // A tree guide for session comparisons: a vertical line through the group, with a branch to each row.
    const renderComparisonGuide = (
      isLast: boolean,
      { width, showBranch = true, verticalPadding }: { width: number; showBranch?: boolean; verticalPadding: number },
    ) => (
      <div
        aria-hidden
        css={{
          position: 'relative',
          alignSelf: 'stretch',
          width,
          flexShrink: 0,
          '&::before': {
            content: "''",
            position: 'absolute',
            top: -verticalPadding,
            bottom: isLast ? '50%' : -verticalPadding,
            left: theme.general.heightSm / 2,
            borderLeft: `1px solid ${theme.colors.borderDecorative}`,
          },
          '&::after': {
            content: showBranch ? "''" : undefined,
            position: 'absolute',
            top: '50%',
            left: theme.general.heightSm / 2,
            width: theme.spacing.sm,
            borderTop: `1px solid ${theme.colors.borderDecorative}`,
          },
        }}
      />
    );

    const renderSessionComparisonRow = (comparison: SessionComparisonRow) => {
      const expanded = expandedComparisonGroups?.has(comparison.id) ?? false;
      const expandedRuns = expandedComparisonRuns?.get(comparison.id);
      const populatedRuns = comparison.runs.filter((run) => run.traces.length > 0);
      const sharedSessionId = populatedRuns[0]?.sessionId;
      const sessionLabel =
        sharedSessionId && populatedRuns.every((run) => run.sessionId === sharedSessionId)
          ? sharedSessionId
          : intl.formatMessage({
              defaultMessage: 'Matched sessions',
              description: 'Label for a comparison group containing different session IDs across runs',
            });
      const turnCount = populatedRuns.reduce((count, run) => count + run.traces.length, 0);
      const guideOptions = {
        width: theme.general.heightSm + theme.spacing.xs,
        verticalPadding: theme.spacing.xs + 2,
      };
      const comparisonRowStyle = { ...rowWidthStyle, minHeight: dataRowStyle.minHeight };
      return (
        <Fragment key={comparison.id}>
          <TableRow
            aria-label={intl.formatMessage(
              {
                defaultMessage: 'Session comparison {sessionId}',
                description: 'Parent row for a matched group of run sessions',
              },
              { sessionId: sessionLabel },
            )}
            style={comparisonRowStyle}
            css={rowPaddingCss}
          >
            <TableCell wrapContent={false} style={{ flex: 1 }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                <Button
                  componentId={`${COMPONENT_ID}.comparison-session-toggle`}
                  size="small"
                  icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                  aria-expanded={expanded}
                  aria-label={
                    expanded
                      ? intl.formatMessage(
                          {
                            defaultMessage: 'Collapse session comparison {sessionId}',
                            description: 'Collapse the run rows within a matched session comparison',
                          },
                          { sessionId: sessionLabel },
                        )
                      : intl.formatMessage(
                          {
                            defaultMessage: 'Expand session comparison {sessionId}',
                            description: 'Expand the run rows within a matched session comparison',
                          },
                          { sessionId: sessionLabel },
                        )
                  }
                  onClick={() => onToggleComparisonGroup?.(comparison.id)}
                />
                <Typography.Text bold ellipsis title={sessionLabel}>
                  {sessionLabel}
                </Typography.Text>
                <Typography.Text size="sm" color="secondary">
                  {intl.formatMessage(
                    {
                      defaultMessage: '{runCount} · {turnCount}',
                      description: 'Number of populated runs and loaded trace turns in a session comparison group',
                    },
                    {
                      runCount: intl.formatMessage(
                        {
                          defaultMessage: '{count, plural, one {# run} other {# runs}}',
                          description: 'Number of populated runs in a session comparison group',
                        },
                        { count: populatedRuns.length },
                      ),
                      turnCount: intl.formatMessage(
                        {
                          defaultMessage: '{count, plural, one {# turn} other {# turns}}',
                          description: 'Number of loaded trace turns across runs in a session comparison group',
                        },
                        { count: turnCount },
                      ),
                    },
                  )}
                </Typography.Text>
              </div>
            </TableCell>
          </TableRow>
          {expanded &&
            comparison.runs.map((run, runIndex) => {
              const first = run.traces[0];
              const last = run.traces.at(-1);
              const firstRow = first ? table.getRow(comparisonSlotId(first.trace_id, run.runUuid)) : undefined;
              const lastRow = last ? table.getRow(comparisonSlotId(last.trace_id, run.runUuid)) : undefined;
              const runExpanded = Boolean(first && expandedRuns?.has(run.runUuid));
              const isLastRun = runIndex === comparison.runs.length - 1;
              const runIdentity = (
                <div css={{ display: 'flex', alignItems: 'center', alignSelf: 'stretch', minWidth: 0 }}>
                  {renderComparisonGuide(isLastRun, guideOptions)}
                  <div css={{ width: theme.general.heightSm, flexShrink: 0 }}>
                    {first && (
                      <Button
                        componentId={`${COMPONENT_ID}.comparison-session-run-toggle`}
                        size="small"
                        icon={runExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                        aria-expanded={runExpanded}
                        aria-label={
                          runExpanded
                            ? intl.formatMessage(
                                {
                                  defaultMessage: 'Collapse {runName} session {sessionId}',
                                  description: 'Collapse one run’s turns within a matched session comparison',
                                },
                                { runName: run.label, sessionId: run.sessionId ?? sessionLabel },
                              )
                            : intl.formatMessage(
                                {
                                  defaultMessage: 'Expand {runName} session {sessionId}',
                                  description: 'Expand one run’s turns within a matched session comparison',
                                },
                                { runName: run.label, sessionId: run.sessionId ?? sessionLabel },
                              )
                        }
                        onClick={() => onToggleComparisonRun?.(comparison.id, run.runUuid)}
                      />
                    )}
                  </div>
                  <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                    <Tooltip componentId={`${COMPONENT_ID}.comparison-session-run-tooltip`} content={run.label}>
                      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                        <span
                          aria-hidden
                          css={{
                            backgroundColor: run.color,
                            width: theme.spacing.sm,
                            height: theme.spacing.sm,
                            borderRadius: '50%',
                            flexShrink: 0,
                          }}
                        />
                        <Typography.Text ellipsis>{run.label}</Typography.Text>
                      </div>
                    </Tooltip>
                    {first && run.sessionId ? (
                      renderSessionHeaderCell(run.sessionId, first)
                    ) : (
                      <Typography.Text color="secondary">—</Typography.Text>
                    )}
                  </div>
                </div>
              );
              return (
                <Fragment key={run.runUuid}>
                  <TableRow
                    aria-label={
                      run.sessionId
                        ? intl.formatMessage(
                            {
                              defaultMessage: '{runName} session {sessionId}',
                              description: 'Summary row for one run in a matched session comparison',
                            },
                            { runName: run.label, sessionId: run.sessionId },
                          )
                        : intl.formatMessage(
                            {
                              defaultMessage: '{runName} session unavailable',
                              description: 'Placeholder row for a run with no matching session',
                            },
                            { runName: run.label },
                          )
                    }
                    style={comparisonRowStyle}
                    css={rowPaddingCss}
                  >
                    {leafHeaders.map((header) => {
                      const sourceRow = header.column.id === 'state' ? lastRow : firstRow;
                      const cell = sourceRow
                        ?.getVisibleCells()
                        .find((candidate) => candidate.column.id === header.column.id);
                      return (
                        <TableCell key={header.id} wrapContent={false} style={columnStyles.get(header.column.id)}>
                          {header.column.id === 'session' ? (
                            runIdentity
                          ) : (
                            <div css={{ display: 'flex', alignItems: 'center', minWidth: 0 }}>
                              {!first ? (
                                <Typography.Text color="secondary">—</Typography.Text>
                              ) : header.column.id === 'input' ? (
                                renderSessionPreview(getTraceInfoInputPreview(first), 'secondary')
                              ) : header.column.id === 'output' ? (
                                renderSessionPreview(getTraceInfoOutputs(last ?? first))
                              ) : (header.column.id === 'start_time' || header.column.id === 'state') && cell ? (
                                flexRender(cell.column.columnDef.cell, cell.getContext())
                              ) : (
                                (sessionCellRenderers.get(header.column.id)?.(run.traces) ?? (
                                  <Typography.Text color="secondary">—</Typography.Text>
                                ))
                              )}
                            </div>
                          )}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                  {runExpanded && (
                    <div
                      role="rowgroup"
                      aria-label={intl.formatMessage(
                        {
                          defaultMessage: '{runName} session turns',
                          description: 'Trace block for one run in an expanded session comparison',
                        },
                        { runName: run.label },
                      )}
                    >
                      {run.traces.map((trace, turn) => {
                        const row = table.getRow(comparisonSlotId(trace.trace_id, run.runUuid));
                        return (
                          <TableRow
                            key={row.id}
                            style={comparisonRowStyle}
                            css={{
                              ...rowPaddingCss,
                              cursor: 'pointer',
                              backgroundColor:
                                rowIdFor(trace) === selectedTraceId
                                  ? theme.colors.tableBackgroundUnselectedHover
                                  : undefined,
                            }}
                            onClick={(event) => {
                              // Cell activators already open the trace; only plain row content needs this fallback.
                              if (
                                event.target instanceof Element &&
                                event.target.closest('a, button, [role="button"]')
                              ) {
                                return;
                              }
                              const href = getTraceHref?.(trace);
                              if ((event.ctrlKey || event.metaKey) && href) {
                                event.preventDefault();
                                window.open(
                                  typeof href === 'string' ? href : createPath(href),
                                  '_blank',
                                  'noopener,noreferrer',
                                );
                                return;
                              }
                              onTraceSelected(trace);
                            }}
                          >
                            {row.getVisibleCells().map((cell) => (
                              <TableCell
                                key={cell.id}
                                wrapContent={cell.column.id !== 'session'}
                                style={columnStyles.get(cell.column.id)}
                              >
                                {cell.column.id === 'session' ? (
                                  <div css={{ display: 'flex', alignItems: 'center', alignSelf: 'stretch' }}>
                                    {isLastRun ? (
                                      <div aria-hidden css={{ width: guideOptions.width, flexShrink: 0 }} />
                                    ) : (
                                      renderComparisonGuide(false, { ...guideOptions, showBranch: false })
                                    )}
                                    {renderComparisonGuide(turn === run.traces.length - 1, guideOptions)}
                                    <Tag componentId={`${COMPONENT_ID}.comparison-session-turn`}>
                                      {intl.formatMessage(
                                        {
                                          defaultMessage: 'Turn {turnNumber}',
                                          description: 'Turn number within one run session, not a cross-run match',
                                        },
                                        { turnNumber: turn + 1 },
                                      )}
                                    </Tag>
                                  </div>
                                ) : (
                                  flexRender(cell.column.columnDef.cell, cell.getContext())
                                )}
                              </TableCell>
                            ))}
                          </TableRow>
                        );
                      })}
                    </div>
                  )}
                </Fragment>
              );
            })}
        </Fragment>
      );
    };

    const renderComparisonTable = (rows: TracesComparisonRow[]) => (
      <div
        role="region"
        aria-busy={isFetching}
        style={tableSizeVariables}
        aria-label={intl.formatMessage({
          defaultMessage: 'Traces',
          description: 'Region label wrapping the traces table',
        })}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          '[role="cell"], [role="cell"] *': { fontSize: `${theme.typography.fontSizeBase}px !important` },
        }}
      >
        <Table scrollable size="default" css={{ flex: 1, isolation: 'isolate' }}>
          <TableRow
            isHeader
            css={comparisonHeaderRowCss}
            // Above the pinned Input cells, which would otherwise show through the sticky header.
            style={{ ...rowWidthStyle, zIndex: theme.options.zIndexBase + 1 }}
          >
            {comparisonInputHeader && renderComparisonHeader(comparisonInputHeader)}
            {comparisonRunHeader && (
              <TableHeader
                componentId={`${COMPONENT_ID}.comparison-run`}
                header={comparisonRunHeader}
                column={comparisonRunHeader.column}
                setColumnSizing={table.setColumnSizing}
                style={columnStyles.get(comparisonRunHeader.column.id)}
                wrapContent={false}
              >
                <TraceColumnHeader
                  columnId={COMPARISON_RUN_COLUMN_ID}
                  label={intl.formatMessage({
                    defaultMessage: 'Run',
                    description: 'Run label column in a trace comparison',
                  })}
                  sortable={false}
                  sortDirection="none"
                  onSortAscending={noop}
                  onSortDescending={noop}
                  titleMinHeight={hasSummaryHeaders ? theme.general.heightSm : undefined}
                />
              </TableHeader>
            )}
            {comparisonValueHeaders.map(renderComparisonHeader)}
          </TableRow>
          {isLoading
            ? Array.from({ length: skeletonRowCount }, (_, i) => (
                <TableRow key={`skeleton-${i}`} css={rowPaddingCss} style={dataRowStyle}>
                  {leafHeaders.map((header) => (
                    <TableCell
                      key={header.id}
                      css={{ verticalAlign: 'middle' }}
                      style={{
                        ...columnStyles.get(header.column.id),
                        ...(header.column.id === 'input' && !isSessionComparison
                          ? pinnedComparisonInputStyle
                          : undefined),
                      }}
                    >
                      <TableSkeleton seed={`traces-${header.id}-${i}`} />
                    </TableCell>
                  ))}
                </TableRow>
              ))
            : rows.map((row) => (row.kind === 'session' ? renderSessionComparisonRow(row) : renderComparisonRow(row)))}
        </Table>
      </div>
    );

    if (comparisonRows) {
      return renderComparisonTable(comparisonRows);
    }

    return (
      <div
        role="region"
        aria-busy={isFetching}
        style={tableSizeVariables}
        aria-label={intl.formatMessage({
          defaultMessage: 'Traces',
          description: 'Region label wrapping the traces table',
        })}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          // Keep every cell at the base font size so density affects row height alone: the DS
          // typography class carries a theme-specific prefix (`du-bois-light-`/`du-bois-dark-`), so
          // pin the whole cell subtree rather than fighting individual styles.
          '[role="cell"], [role="cell"] *': { fontSize: `${theme.typography.fontSizeBase}px !important` },
        }}
      >
        {/* `scrollable` makes the DuBois Table the scroll container (both axes), which is what activates
          its sticky-header CSS; `flex: 1` gives it a bounded height from the flex parent to scroll within.
          The DndContext wraps the whole table so the drag overlay can render over the rows when column
          reordering is enabled. */}
        {onReorderColumn ? (
          <DndContext
            sensors={headerDragSensors}
            collisionDetection={pointerXAxisCollisionDetection}
            onDragStart={handleHeaderDragStart}
            onDragEnd={handleHeaderDragEnd}
            onDragCancel={() => setActiveDragColumnId(null)}
          >
            <Table
              scrollable
              size="default"
              css={{ flex: 1 }}
              someRowsSelected={isAllOnPageSelected || isSomeOnPageSelected}
            >
              <TableRow isHeader css={headerRowCss} style={headerRowStyle}>
                <TableRowSelectCell
                  componentId={`${COMPONENT_ID}.row-select-all`}
                  checked={isAllOnPageSelected}
                  indeterminate={isSomeOnPageSelected && !isAllOnPageSelected}
                  onChange={onToggleBulkAll}
                  checkboxLabel={intl.formatMessage({
                    defaultMessage: 'Select all traces on this page',
                    description: 'Aria label for the select-all checkbox in the traces table header',
                  })}
                />
                {isGroupedBySession && renderSessionToggleSpacer()}
                <SortableContext items={reorderableHeaderIds} strategy={horizontalListSortingStrategy}>
                  {leafHeaders.map((header) => {
                    const columnId = header.column.id;
                    const labelNode = flexRender(header.column.columnDef.header, header.getContext());
                    // Sort lives in the menu, so no DuBois `sortable` (its button wrapper can't nest the trigger).
                    const sortHandlers = isSortableTraceColumn(columnId)
                      ? {
                          onSortAscending: () => onSort(columnId, 'asc'),
                          onSortDescending: () => onSort(columnId, 'desc'),
                        }
                      : { onSortAscending: noop, onSortDescending: noop };
                    // Prefer explicit labelText on the column def (for a11y on JSX headers), fall back to string headers.
                    const columnDef = header.column.columnDef as TraceTableColumn;
                    const labelText = columnDef.labelText ?? (typeof labelNode === 'string' ? labelNode : undefined);
                    return (
                      <SortableTraceTableHeader
                        key={header.id}
                        columnId={columnId}
                        componentId={`${COMPONENT_ID}.header`}
                        header={header}
                        column={header.column}
                        setColumnSizing={table.setColumnSizing}
                        style={columnStyles.get(columnId)}
                        wrapContent={false}
                      >
                        <TraceColumnHeader
                          columnId={columnId}
                          label={labelNode}
                          labelText={labelText}
                          sortable={isSortableTraceColumn(columnId)}
                          sortDirection={sort === columnId ? dir : 'none'}
                          onHide={() => onHideColumn(columnId)}
                          action={columnHeaderActions?.[columnId]}
                          headerVariant={getTraceColumnHeaderVariant(columnDef)}
                          {...sortHandlers}
                        />
                      </SortableTraceTableHeader>
                    );
                  })}
                </SortableContext>
                {headerBottomOverlay && renderHeaderBottomOverlay()}
              </TableRow>
              {headerBottomOverlay && renderHeaderBottomOverlaySpacer()}

              {isLoading
                ? Array.from({ length: skeletonRowCount }, (_, i) => (
                    <TableRow
                      key={`skeleton-${i}`}
                      css={{ ...rowPaddingCss, ...dataSelectCellAlign }}
                      style={dataRowStyle}
                    >
                      <TableRowSelectCell componentId={`${COMPONENT_ID}.row-select.skeleton`} noCheckbox />
                      {isGroupedBySession && renderSessionToggleSpacer()}
                      {leafHeaders.map((header) => (
                        <TableCell
                          key={header.id}
                          css={{ verticalAlign: 'middle' }}
                          style={columnStyles.get(header.column.id)}
                        >
                          <TableSkeleton seed={`traces-${header.id}-${i}`} />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                : groupedRows
                  ? groupedRows.map(({ sessionId, rows }) => {
                      // A standalone trace (no session id) renders as an ordinary row, indented under the
                      // toggle column so it lines up with the session rows' content.
                      if (!sessionId) {
                        return renderTraceRow(rows[0], true);
                      }
                      const isExpanded = expandedSessions.has(sessionId);
                      const tracesInSession = rows.map((row) => row.original);
                      const selectedCount = tracesInSession.filter((trace) =>
                        selectedForBulk.has(trace.trace_id),
                      ).length;
                      return (
                        <Fragment key={sessionId}>
                          <TableRow
                            isHeader
                            style={sessionRowStyle}
                            css={{
                              cursor: onSessionSelected ? 'pointer' : undefined,
                              ...rowPaddingCss,
                              ...dataSelectCellAlign,
                            }}
                            onClick={
                              onSessionSelected
                                ? () => onSessionSelected({ trace: rows[0].original, sessionId })
                                : undefined
                            }
                          >
                            <TableRowSelectCell
                              componentId={`${COMPONENT_ID}.session-select`}
                              checked={selectedCount === tracesInSession.length}
                              indeterminate={selectedCount > 0 && selectedCount < tracesInSession.length}
                              isDisabled={!onToggleBulkRows}
                              onChange={() => onToggleBulkRows?.(tracesInSession)}
                              {...stopPropagationProps}
                              checkboxLabel={intl.formatMessage(
                                {
                                  defaultMessage: 'Select session {sessionId}',
                                  description: 'Aria label for selecting every trace in a grouped session',
                                },
                                { sessionId },
                              )}
                            />
                            <div css={{ width: sessionToggleWidth, flexShrink: 0 }}>
                              <Button
                                componentId={`${COMPONENT_ID}.session-toggle`}
                                size="small"
                                icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                                aria-label={
                                  isExpanded
                                    ? intl.formatMessage(
                                        {
                                          defaultMessage: 'Collapse session {sessionId}',
                                          description: 'Accessible label for collapsing a grouped session',
                                        },
                                        { sessionId },
                                      )
                                    : intl.formatMessage(
                                        {
                                          defaultMessage: 'Expand session {sessionId}',
                                          description: 'Accessible label for expanding a grouped session',
                                        },
                                        { sessionId },
                                      )
                                }
                                onClick={(event) => {
                                  event.stopPropagation();
                                  toggleSessionExpanded(sessionId);
                                }}
                              />
                            </div>
                            {leafHeaders.map((header) => {
                              const firstCell = rows[0]?.getVisibleCells().find((cell) => cell.column.id === header.id);
                              const lastCell = rows
                                .at(-1)
                                ?.getVisibleCells()
                                .find((cell) => cell.column.id === header.id);
                              return (
                                <TableCell
                                  key={header.id}
                                  css={{ verticalAlign: 'middle' }}
                                  style={columnStyles.get(header.column.id)}
                                >
                                  {/* Session summary per column: the session tag, first-turn input, last-turn
                                      output/state, first-turn time, else a product-owned aggregate (or blank). */}
                                  {header.column.id === 'session'
                                    ? renderSessionHeaderCell(sessionId, rows[0].original)
                                    : header.column.id === 'input'
                                      ? renderSessionPreview(getTraceInfoInputs(rows[0].original), 'secondary')
                                      : header.column.id === 'output'
                                        ? renderSessionPreview(
                                            getTraceInfoOutputs(rows.at(-1)?.original ?? rows[0].original),
                                          )
                                        : header.column.id === 'start_time' && firstCell
                                          ? flexRender(firstCell.column.columnDef.cell, firstCell.getContext())
                                          : header.column.id === 'state' && lastCell
                                            ? flexRender(lastCell.column.columnDef.cell, lastCell.getContext())
                                            : // Suppress product-owned aggregates when the page may not
                                              // contain all traces for this session — a partial total is
                                              // more confusing than a blank. Mirrors onToggleBulkRows.
                                              sessionsMayBeIncomplete
                                              ? null
                                              : (sessionCellRenderers.get(header.column.id)?.(tracesInSession) ?? null)}
                                </TableCell>
                              );
                            })}
                          </TableRow>
                          {isExpanded && rows.map((row, index) => renderTraceRow(row, true, index + 1))}
                        </Fragment>
                      );
                    })
                  : table.getRowModel().rows.map((row) => renderTraceRow(row))}
            </Table>
            {createPortal(
              <DragOverlay
                modifiers={HEADER_DRAG_OVERLAY_MODIFIERS}
                style={{ width: HEADER_DRAG_OVERLAY_WIDTH, height: HEADER_DRAG_OVERLAY_HEIGHT }}
                zIndex={theme.options.zIndexBase + 1}
              >
                {activeDragHeaderLabel ? <TraceTableHeaderDragOverlay label={activeDragHeaderLabel} /> : null}
              </DragOverlay>,
              document.body,
            )}
          </DndContext>
        ) : (
          <Table
            scrollable
            size="default"
            css={{ flex: 1 }}
            someRowsSelected={isAllOnPageSelected || isSomeOnPageSelected}
          >
            <TableRow isHeader css={headerRowCss} style={headerRowStyle}>
              <TableRowSelectCell
                componentId={`${COMPONENT_ID}.row-select-all`}
                checked={isAllOnPageSelected}
                indeterminate={isSomeOnPageSelected && !isAllOnPageSelected}
                onChange={onToggleBulkAll}
                checkboxLabel={intl.formatMessage({
                  defaultMessage: 'Select all traces on this page',
                  description: 'Aria label for the select-all checkbox in the traces table header',
                })}
              />
              {isGroupedBySession && renderSessionToggleSpacer()}
              {leafHeaders.map((header) => {
                const columnId = header.column.id;
                const labelNode = flexRender(header.column.columnDef.header, header.getContext());
                // Sort lives in the menu, so no DuBois `sortable` (its button wrapper can't nest the trigger).
                const sortHandlers = isSortableTraceColumn(columnId)
                  ? {
                      onSortAscending: () => onSort(columnId, 'asc'),
                      onSortDescending: () => onSort(columnId, 'desc'),
                    }
                  : { onSortAscending: noop, onSortDescending: noop };
                // Prefer explicit labelText on the column def (for a11y on JSX headers), fall back to string headers.
                const columnDef = header.column.columnDef as TraceTableColumn;
                const labelText = columnDef.labelText ?? (typeof labelNode === 'string' ? labelNode : undefined);
                return (
                  <TableHeader
                    key={header.id}
                    componentId={`${COMPONENT_ID}.header`}
                    header={header}
                    column={header.column}
                    setColumnSizing={table.setColumnSizing}
                    style={columnStyles.get(columnId)}
                    wrapContent={false}
                  >
                    <TraceColumnHeader
                      columnId={columnId}
                      label={labelNode}
                      labelText={labelText}
                      sortable={isSortableTraceColumn(columnId)}
                      sortDirection={sort === columnId ? dir : 'none'}
                      onHide={() => onHideColumn(columnId)}
                      action={columnHeaderActions?.[columnId]}
                      headerVariant={getTraceColumnHeaderVariant(columnDef)}
                      {...sortHandlers}
                    />
                  </TableHeader>
                );
              })}
              {headerBottomOverlay && renderHeaderBottomOverlay()}
            </TableRow>
            {headerBottomOverlay && renderHeaderBottomOverlaySpacer()}

            {isLoading
              ? Array.from({ length: skeletonRowCount }, (_, i) => (
                  <TableRow
                    key={`skeleton-${i}`}
                    css={{ ...rowPaddingCss, ...dataSelectCellAlign }}
                    style={dataRowStyle}
                  >
                    <TableRowSelectCell componentId={`${COMPONENT_ID}.row-select.skeleton`} noCheckbox />
                    {isGroupedBySession && renderSessionToggleSpacer()}
                    {leafHeaders.map((header) => (
                      <TableCell
                        key={header.id}
                        css={{ verticalAlign: 'middle' }}
                        style={columnStyles.get(header.column.id)}
                      >
                        <TableSkeleton seed={`traces-${header.id}-${i}`} />
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              : groupedRows
                ? groupedRows.map(({ sessionId, rows }) => {
                    // A standalone trace (no session id) renders as an ordinary row, indented under the
                    // toggle column so it lines up with the session rows' content.
                    if (!sessionId) {
                      return renderTraceRow(rows[0], true);
                    }
                    const isExpanded = expandedSessions.has(sessionId);
                    const tracesInSession = rows.map((row) => row.original);
                    const selectedCount = tracesInSession.filter((trace) => selectedForBulk.has(trace.trace_id)).length;
                    return (
                      <Fragment key={sessionId}>
                        <TableRow
                          isHeader
                          style={sessionRowStyle}
                          css={{
                            cursor: onSessionSelected ? 'pointer' : undefined,
                            ...rowPaddingCss,
                            ...dataSelectCellAlign,
                          }}
                          onClick={
                            onSessionSelected
                              ? () => onSessionSelected({ trace: rows[0].original, sessionId })
                              : undefined
                          }
                        >
                          <TableRowSelectCell
                            componentId={`${COMPONENT_ID}.session-select`}
                            checked={selectedCount === tracesInSession.length}
                            indeterminate={selectedCount > 0 && selectedCount < tracesInSession.length}
                            isDisabled={!onToggleBulkRows}
                            onChange={() => onToggleBulkRows?.(tracesInSession)}
                            {...stopPropagationProps}
                            checkboxLabel={intl.formatMessage(
                              {
                                defaultMessage: 'Select session {sessionId}',
                                description: 'Aria label for selecting every trace in a grouped session',
                              },
                              { sessionId },
                            )}
                          />
                          <div css={{ width: sessionToggleWidth, flexShrink: 0 }}>
                            <Button
                              componentId={`${COMPONENT_ID}.session-toggle`}
                              size="small"
                              icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                              aria-label={
                                isExpanded
                                  ? intl.formatMessage(
                                      {
                                        defaultMessage: 'Collapse session {sessionId}',
                                        description: 'Accessible label for collapsing a grouped session',
                                      },
                                      { sessionId },
                                    )
                                  : intl.formatMessage(
                                      {
                                        defaultMessage: 'Expand session {sessionId}',
                                        description: 'Accessible label for expanding a grouped session',
                                      },
                                      { sessionId },
                                    )
                              }
                              onClick={(event) => {
                                event.stopPropagation();
                                toggleSessionExpanded(sessionId);
                              }}
                            />
                          </div>
                          {leafHeaders.map((header) => {
                            const firstCell = rows[0]?.getVisibleCells().find((cell) => cell.column.id === header.id);
                            const lastCell = rows
                              .at(-1)
                              ?.getVisibleCells()
                              .find((cell) => cell.column.id === header.id);
                            return (
                              <TableCell
                                key={header.id}
                                css={{ verticalAlign: 'middle' }}
                                style={columnStyles.get(header.column.id)}
                              >
                                {/* Session summary per column: the session tag, first-turn input, last-turn
                                    output/state, first-turn time, else a product-owned aggregate (or blank). */}
                                {header.column.id === 'session'
                                  ? renderSessionHeaderCell(sessionId, rows[0].original)
                                  : header.column.id === 'input'
                                    ? renderSessionPreview(getTraceInfoInputs(rows[0].original), 'secondary')
                                    : header.column.id === 'output'
                                      ? renderSessionPreview(
                                          getTraceInfoOutputs(rows.at(-1)?.original ?? rows[0].original),
                                        )
                                      : header.column.id === 'start_time' && firstCell
                                        ? flexRender(firstCell.column.columnDef.cell, firstCell.getContext())
                                        : header.column.id === 'state' && lastCell
                                          ? flexRender(lastCell.column.columnDef.cell, lastCell.getContext())
                                          : sessionsMayBeIncomplete
                                            ? null
                                            : (sessionCellRenderers.get(header.column.id)?.(tracesInSession) ?? null)}
                              </TableCell>
                            );
                          })}
                        </TableRow>
                        {isExpanded && rows.map((row, index) => renderTraceRow(row, true, index + 1))}
                      </Fragment>
                    );
                  })
                : table.getRowModel().rows.map((row) => renderTraceRow(row))}
          </Table>
        )}
      </div>
    );
  },
);

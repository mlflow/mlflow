import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  SpeechBubbleIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeleton,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
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
import { getTraceInfoInputs, getTraceInfoOutputs } from '../genai-traces-table/utils/TraceUtils';
import { Link } from '../genai-traces-table/utils/RoutingUtils';
import { type ColumnSizingState, flexRender, getCoreRowModel, type Row } from '@tanstack/react-table';
import { Fragment, memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { createPath } from 'react-router';
import { isSortableTraceColumn } from './constants';
import type {
  SessionHrefGetter,
  SessionSelectionHandler,
  SortDirection,
  TraceColumnId,
  TraceColumnHeaderAction,
  TraceHrefGetter,
  TraceTableColumn,
} from './types';
import { getVisibleColumnDefs, type TracesTableMeta } from './columns';
import { getContentColumnMaxSizes } from './getColumnMaxSizes';
import { TraceColumnHeader } from './TraceColumnHeader';

// No-op sort handlers for non-sortable columns (their header menu omits the sort items anyway).
const noop = () => {};

// Module-local static analytics-id namespace (the `@databricks/no-dynamic-property-value` rule
// requires `componentId` values to be static, so a runtime-injected prefix isn't possible).
const COMPONENT_ID = 'web-shared.traces-table';

// Input/Output are the fill columns: each takes grow factor 1 so any leftover container width is
// split equally between them (no dead whitespace on the right), with `maxWidth: unset` so the cap
// can't block that growth. Every other column stays fixed-width — it occupies exactly its (possibly
// resized) pixel width and horizontal scroll on the Table handles overflow. A drag still sets
// `columnSizing[id]`, which becomes the new flex-basis.
const GROWING_COLUMNS = new Set<string>(['input', 'output']);
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

export interface TracesTableProps {
  traces: ModelTraceInfoV3[];
  visibleColumns: TraceColumnId[];
  /** Product-specific columns appended after the standard columns. Memoize a stable reference. */
  extraColumns?: TraceTableColumn[];
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
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
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
  columnHeaderActions?: Readonly<Partial<Record<string, TraceColumnHeaderAction>>>;
  /** Groups traces with a session id into collapsible session rows. Standalone traces remain rows. */
  isGroupedBySession?: boolean;
  /** Maximum lines shown by input and output previews before truncation. Defaults to one line. */
  previewLineClamp?: number;
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
    visibleColumns,
    extraColumns,
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
    getTraceHref,
    getSessionHref,
    onSessionSelected,
    onFilterByTag,
    renderRunName,
    onHideColumn,
    columnHeaderActions,
    isGroupedBySession = false,
    previewLineClamp = 1,
  }: TracesTableProps) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    // Keep the checkbox centered and leave a full spacing token before the first data column.
    const selectCellCss = {
      '.table-row-select-cell': { alignItems: 'center', paddingRight: theme.spacing.sm },
    } as const;

    // Canonical-order visible column defs + any product columns. `extraColumns` is guarded to a stable
    // reference so a stable/undefined value doesn't defeat the deep memo (see `getVisibleColumnDefs`).
    const columns = useMemo(() => {
      // Grouped mode always shows session + the input/output previews and pins them left, so a session
      // header reads left-to-right as "which session → first input → last output" regardless of the
      // user's own column visibility/order.
      const groupedLeadingColumns: TraceColumnId[] = ['session', 'input', 'output'];
      const groupedVisibleColumns = isGroupedBySession
        ? [...new Set([...visibleColumns, ...groupedLeadingColumns])]
        : visibleColumns;
      const visibleColumnDefs = getVisibleColumnDefs(groupedVisibleColumns, extraColumns);
      const groupedColumnRank = new Map<string, number>(groupedLeadingColumns.map((id, index) => [id, index]));
      const getGroupedColumnRank = (id: string | undefined) =>
        id === undefined ? Infinity : (groupedColumnRank.get(id) ?? Infinity);
      const orderedVisibleColumnDefs = isGroupedBySession
        ? [...visibleColumnDefs].sort((left, right) => getGroupedColumnRank(left.id) - getGroupedColumnRank(right.id))
        : visibleColumnDefs;
      // Product-specific column ids are intentionally absent and resolve to undefined.
      const contentMaxSizes: Readonly<Record<string, number | undefined>> = getContentColumnMaxSizes(traces, intl);
      return orderedVisibleColumnDefs.map((column) => {
        if (column.id === undefined) {
          return column;
        }
        const contentMaxSize = contentMaxSizes[column.id];
        const persistedSize = initialColumnSizing[column.id];
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
    }, [visibleColumns, extraColumns, isGroupedBySession, traces, intl, initialColumnSizing]);

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

    const meta = useMemo<TracesTableMeta>(
      () => ({ intl, onTraceSelected, getTraceHref, getSessionHref, onFilterByTag, renderRunName, previewLineClamp }),
      [intl, onTraceSelected, getTraceHref, getSessionHref, onFilterByTag, renderRunName, previewLineClamp],
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

    const table = useReactTable_unverifiedWithReact18<ModelTraceInfoV3>('traces-table/TracesTable.tsx', {
      data: traces,
      columns,
      getCoreRowModel: getCoreRowModel(),
      getRowId: rowIdFor,
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
    const dataRowStyle: CSSProperties = rowMinHeight ? { ...rowWidthStyle, minHeight: rowMinHeight } : rowWidthStyle;

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
    const selectCellWidth = theme.spacing.md + theme.spacing.sm * 2;
    // Grouped mode inserts a leading expand/collapse toggle cell (a small button) before the columns;
    // reserve its width so header, session, and trace rows all stay column-aligned.
    const sessionToggleWidth = isGroupedBySession ? theme.general.heightSm + theme.spacing.xs : 0;
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

    // Header-row overrides: align its contents and set the header's bottom border.
    const headerRowCss = {
      ...selectCellCss,
      borderBottom: `1px solid ${theme.colors.border}`,
      // Center each header label against the (vertically-centered) select-all checkbox. Padding sets
      // the label→divider gap directly — DS applies --table-row-vertical-padding via an inline style,
      // so a `css` override of it is a no-op.
      '[role="columnheader"]': {
        alignItems: 'center',
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
          its sticky-header CSS; `flex: 1` gives it a bounded height from the flex parent to scroll within. */}
        <Table
          scrollable
          size="default"
          css={{ flex: 1 }}
          someRowsSelected={isAllOnPageSelected || isSomeOnPageSelected}
        >
          <TableRow isHeader css={headerRowCss} style={rowWidthStyle}>
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
                    {...sortHandlers}
                  />
                </TableHeader>
              );
            })}
          </TableRow>

          {isLoading
            ? Array.from({ length: skeletonRowCount }, (_, i) => (
                <TableRow key={`skeleton-${i}`} css={{ ...rowPaddingCss, ...dataSelectCellAlign }} style={dataRowStyle}>
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
                        style={rowWidthStyle}
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
      </div>
    );
  },
);

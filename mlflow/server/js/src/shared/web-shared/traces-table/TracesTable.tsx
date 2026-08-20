import {
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeleton,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
// This react-table package predates `useReactTableWithDeepMemo`. Use the canonical
// `useReactTable_unverifiedWithReact18` (the same hook `GenAiTracesTableBody` uses for react-query
// data): `data` comes reference-stable from `useTracesPageQuery`, and `columns`/`meta` are memoized
// below, so no deep-equality guard is needed.
import { useReactTable_unverifiedWithReact18 } from '../react-table/useReactTable';
import { createTraceV4LongIdentifier } from '../model-trace-explorer/ModelTraceExplorer.utils';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { doesTraceSupportV4API } from '../genai-traces-table/utils/TraceLocationUtils';
import { type ColumnSizingState, flexRender, getCoreRowModel } from '@tanstack/react-table';
import { memo, useEffect, useMemo, useRef } from 'react';
import { isSortableTraceColumn } from './constants';
import type { SessionHrefGetter, SortDirection, TraceColumnId, TraceTableColumn } from './types';
import { getVisibleColumnDefs, type TracesTableMeta } from './columns';
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
const sizeStyleFor = (id: string, width: number) =>
  GROWING_COLUMNS.has(id) ? { flex: `1 0 ${width}px`, maxWidth: 'unset' } : { flex: `0 0 ${width}px`, maxWidth: width };

// Row click opens the drawer; the checkbox cell must not also trigger it.
const stopPropagationProps = {
  onClick: (event: React.MouseEvent) => event.stopPropagation(),
};
// Center the row-select checkbox vertically against the single-line cell text.
const selectCellAlign = { '.table-row-select-cell': { alignItems: 'center' } } as const;

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
  onToggleBulkRow: (trace: ModelTraceInfoV3) => void;
  onToggleBulkAll: () => void;
  sort: TraceColumnId;
  dir: SortDirection;
  onSort: (column: TraceColumnId, direction: SortDirection) => void;
  /** Resolves the session cell's link destination; when absent the session renders as plain text. */
  getSessionHref?: SessionHrefGetter;
  /** Toggle a tag filter — wired to the tag pills in the Tags cell; absent → non-clickable pills. */
  onFilterByTag?: (key: string, value: string) => void;
  /** Product-owned renderer for resolving an experiment-scoped run name. */
  renderRunName?: (trace: ModelTraceInfoV3) => React.ReactNode;
  /** Hides the column with the given id — wired to the per-header menu's "Hide column" item. */
  onHideColumn: (columnId: string) => void;
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
    onToggleBulkAll,
    sort,
    dir,
    onSort,
    getSessionHref,
    onFilterByTag,
    renderRunName,
    onHideColumn,
  }: TracesTableProps) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();

    // Canonical-order visible column defs + any product columns. `extraColumns` is guarded to a stable
    // reference so a stable/undefined value doesn't defeat the deep memo (see `getVisibleColumnDefs`).
    const columns = useMemo(() => getVisibleColumnDefs(visibleColumns, extraColumns), [visibleColumns, extraColumns]);

    const meta = useMemo<TracesTableMeta>(
      () => ({ intl, onTraceSelected, getSessionHref, onFilterByTag, renderRunName }),
      [intl, onTraceSelected, getSessionHref, onFilterByTag, renderRunName],
    );

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
      initialState: { columnSizing: initialColumnSizing },
      meta,
    });

    // Persist only on the falling edge of a resize (mouseup): one write per drag, none on mount. The
    // falling-edge ref avoids the redundant mount write that a plain `if (!isResizing) persist(...)`
    // would do.
    const isResizingColumn = Boolean(table.getState().columnSizingInfo.isResizingColumn);
    const columnSizing = table.getState().columnSizing;
    const wasResizing = useRef(false);
    useEffect(() => {
      if (wasResizing.current && !isResizingColumn) {
        onColumnSizingSettled(columnSizing);
      }
      wasResizing.current = isResizingColumn;
    }, [isResizingColumn, columnSizing, onColumnSizingSettled]);

    const leafHeaders = table.getLeafHeaders();

    // Pin every row to the summed width of the visible columns so its hover/selected background spans
    // the full horizontal extent, not just the visible viewport. A DuBois `scrollable` Table makes the
    // table its own horizontal scroll container, and each `TableRow` is a `display: flex` block with no
    // explicit width — so it's only as wide as the scroll *viewport*. The cells are `flex-shrink: 0`, so
    // once the columns overflow they extend past the row box and the row background (which paints only to
    // the row's own width) cuts off at the viewport's right edge. `minWidth: 100%` keeps rows filling the
    // viewport — and lets the input/output fill columns grow — when the columns are narrower than it.
    //
    // The leading select cell lives outside the TanStack column set: it's content-box with a 16px
    // (spacing.md) content width + 8px (spacing.sm) left padding, so it contributes a fixed 24px.
    const selectCellWidth = theme.spacing.md + theme.spacing.sm;
    const rowWidth = leafHeaders.reduce((total, header) => total + header.getSize(), selectCellWidth);
    const rowWidthStyle = { width: rowWidth, minWidth: '100%' } as const;

    // Header-row overrides: near-black labels and a visible divider.
    const headerRowCss = {
      ...selectCellAlign,
      // Center each header label against the (vertically-centered) select-all checkbox. Padding sets
      // the label→divider gap directly — DS applies --table-row-vertical-padding via an inline style,
      // so a `css` override of it is a no-op.
      '[role="columnheader"]': { alignItems: 'center', paddingBottom: theme.spacing.sm },
      // The select-all cell keeps the DS default bottom padding otherwise, which shifts the centered
      // checkbox up off the label line — zero it so the checkbox sits on the labels' center.
      '&& .table-row-select-cell': { paddingBottom: 0 },
      // Doubled `&&` beats the DS 2-class `.table-header-text` var rule, forcing the darkest token.
      '&& .table-header-text': { color: theme.colors.textPrimary },
      // Light divider (grey200), overriding the separator var for the header subtree only.
      ['--table-separator-color' as string]: theme.colors.grey200,
      // Keep the header's select-all checkbox always visible (data rows stay hover-reveal).
      '&& .table-row-select-cell input[type="checkbox"] ~ *': { opacity: 1 },
    };

    return (
      <div
        role="region"
        aria-busy={isFetching}
        aria-label={intl.formatMessage({
          defaultMessage: 'Traces',
          description: 'Region label wrapping the traces table',
        })}
        css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
      >
        {/* `scrollable` makes the DuBois Table the scroll container (both axes), which is what activates
          its sticky-header CSS; `flex: 1` gives it a bounded height from the flex parent to scroll within. */}
        <Table scrollable css={{ flex: 1 }} someRowsSelected={isAllOnPageSelected || isSomeOnPageSelected}>
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
              return (
                <TableHeader
                  key={header.id}
                  componentId={`${COMPONENT_ID}.header`}
                  header={header}
                  column={header.column}
                  setColumnSizing={table.setColumnSizing}
                  style={sizeStyleFor(columnId, header.getSize())}
                  wrapContent={false}
                >
                  <TraceColumnHeader
                    label={labelNode}
                    labelText={typeof labelNode === 'string' ? labelNode : undefined}
                    sortable={isSortableTraceColumn(columnId)}
                    sortDirection={sort === columnId ? dir : 'none'}
                    onHide={() => onHideColumn(columnId)}
                    {...sortHandlers}
                  />
                </TableHeader>
              );
            })}
          </TableRow>

          {isLoading
            ? Array.from({ length: skeletonRowCount }, (_, i) => (
                <TableRow key={`skeleton-${i}`} css={selectCellAlign} style={rowWidthStyle}>
                  <TableRowSelectCell componentId={`${COMPONENT_ID}.row-select.skeleton`} noCheckbox />
                  {leafHeaders.map((header) => (
                    <TableCell
                      key={header.id}
                      css={{ verticalAlign: 'middle' }}
                      style={sizeStyleFor(header.column.id, header.getSize())}
                    >
                      <TableSkeleton seed={`traces-${header.id}-${i}`} />
                    </TableCell>
                  ))}
                </TableRow>
              ))
            : table.getRowModel().rows.map((row) => {
                const isSelected = row.id === selectedTraceId;
                const isBulkChecked = selectedForBulk.has(row.original.trace_id);
                return (
                  <TableRow
                    key={row.id}
                    onClick={() => onTraceSelected(row.original)}
                    style={rowWidthStyle}
                    css={{
                      cursor: 'pointer',
                      backgroundColor: isSelected ? theme.colors.tableBackgroundUnselectedHover : undefined,
                      ...selectCellAlign,
                    }}
                  >
                    <TableRowSelectCell
                      componentId={`${COMPONENT_ID}.row-select`}
                      checked={isBulkChecked}
                      onChange={() => onToggleBulkRow(row.original)}
                      checkboxLabel={intl.formatMessage(
                        {
                          defaultMessage: 'Select trace {traceId}',
                          description: 'Aria label for the per-row select checkbox in the traces table',
                        },
                        { traceId: row.original.trace_id },
                      )}
                      {...stopPropagationProps}
                    />
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        css={{ verticalAlign: 'middle' }}
                        style={sizeStyleFor(cell.column.id, cell.column.getSize())}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })}
        </Table>
      </div>
    );
  },
);

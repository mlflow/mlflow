import { FormattedMessage, type IntlShape } from '@databricks/i18n';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { COLUMN_SIZES } from './constants';
import { TRACE_COLUMN_LABELS } from './columnLabels';
import type { SessionHrefGetter, TraceColumnId, TraceTableColumn } from './types';
import {
  TraceCostCell,
  TraceDurationCell,
  TraceIdCell,
  TraceInputCell,
  TraceNameCell,
  TraceOutputCell,
  TraceRunNameCell,
  TraceSessionCell,
  TraceSourceCell,
  TraceStartTimeCell,
  TraceStateCell,
  TraceTagsCell,
  TraceTokensCell,
  TraceUserCell,
} from './TraceCell';

/**
 * Dynamic params handed to the module-scope cell renderers via the table `meta` option (per the
 * repo table-columns rule — column defs live at module scope, so per-render values arrive here).
 */
export interface TracesTableMeta {
  intl: IntlShape;
  onTraceSelected: (trace: ModelTraceInfoV3) => void;
  /** Resolves the session cell's link destination; when absent the session renders as plain text. */
  getSessionHref?: SessionHrefGetter;
  /** Toggle a tag filter — wired to the tag pills in the Tags cell; absent → non-clickable pills. */
  onFilterByTag?: (key: string, value: string) => void;
  /** Product-owned renderer for resolving an experiment-scoped run name. */
  renderRunName?: (trace: ModelTraceInfoV3) => React.ReactNode;
}

export const getTableMeta = (context: CellContext<ModelTraceInfoV3, unknown>): TracesTableMeta =>
  context.table.options.meta as TracesTableMeta;

// Aria label for a clickable cell that opens the trace drawer. Kept in one place so every cell's
// label reads identically; the first arg to formatMessage stays a static literal (i18n extraction).
export const openLabel = (intl: IntlShape, traceId: string, column: string): string =>
  intl.formatMessage(
    {
      defaultMessage: 'Open trace {traceId} — {column}',
      description: 'Aria label for a clickable traces-table cell that opens the trace detail drawer',
    },
    { traceId, column },
  );

// A column def whose `id` is a known TraceColumnId (TanStack widens `id` to `string`; narrowing it
// here lets the visibility lookups stay cast-free and gives an exhaustiveness check on the list).
type StandardColumnDef = ColumnDef<ModelTraceInfoV3> & { id: TraceColumnId };

/**
 * One ColumnDef per TraceColumnId, at module scope (never rebuilt per render — the load-bearing perf
 * property). Each `cell` reads intl + the callbacks from table `meta`; the header is a plain node
 * (sort affordance is layered on at render time in `TracesTable`). The `id` doubles as the analytics
 * identity via the DS `column` prop on the header.
 */
export const STANDARD_COLUMNS: StandardColumnDef[] = [
  {
    id: 'trace_id',
    ...COLUMN_SIZES.trace_id,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.trace_id} />,
    cell: (ctx) => {
      const { intl, onTraceSelected } = getTableMeta(ctx);
      const trace = ctx.row.original;
      return (
        <TraceIdCell
          trace={trace}
          onSelect={onTraceSelected}
          accessibleLabel={openLabel(intl, trace.trace_id, 'trace id')}
        />
      );
    },
  },
  {
    id: 'trace_name',
    ...COLUMN_SIZES.trace_name,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.trace_name} />,
    cell: (ctx) => <TraceNameCell trace={ctx.row.original} />,
  },
  {
    id: 'start_time',
    ...COLUMN_SIZES.start_time,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.start_time} />,
    cell: (ctx) => <TraceStartTimeCell trace={ctx.row.original} />,
  },
  {
    id: 'input',
    ...COLUMN_SIZES.input,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.input} />,
    cell: (ctx) => {
      const { intl, onTraceSelected } = getTableMeta(ctx);
      const trace = ctx.row.original;
      return (
        <TraceInputCell
          trace={trace}
          onSelect={onTraceSelected}
          accessibleLabel={openLabel(intl, trace.trace_id, 'input')}
        />
      );
    },
  },
  {
    id: 'output',
    ...COLUMN_SIZES.output,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.output} />,
    cell: (ctx) => {
      const { intl, onTraceSelected } = getTableMeta(ctx);
      const trace = ctx.row.original;
      return (
        <TraceOutputCell
          trace={trace}
          onSelect={onTraceSelected}
          accessibleLabel={openLabel(intl, trace.trace_id, 'output')}
        />
      );
    },
  },
  {
    id: 'user',
    ...COLUMN_SIZES.user,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.user} />,
    cell: (ctx) => <TraceUserCell trace={ctx.row.original} />,
  },
  {
    id: 'session',
    ...COLUMN_SIZES.session,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.session} />,
    cell: (ctx) => <TraceSessionCell trace={ctx.row.original} getSessionHref={getTableMeta(ctx).getSessionHref} />,
  },
  {
    id: 'duration',
    ...COLUMN_SIZES.duration,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.duration} />,
    cell: (ctx) => <TraceDurationCell trace={ctx.row.original} />,
  },
  {
    id: 'state',
    ...COLUMN_SIZES.state,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.state} />,
    cell: (ctx) => <TraceStateCell trace={ctx.row.original} />,
  },
  {
    id: 'source',
    ...COLUMN_SIZES.source,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.source} />,
    cell: (ctx) => <TraceSourceCell trace={ctx.row.original} />,
  },
  {
    id: 'run_name',
    ...COLUMN_SIZES.run_name,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.run_name} />,
    cell: (ctx) => <TraceRunNameCell trace={ctx.row.original} renderRunName={getTableMeta(ctx).renderRunName} />,
  },
  {
    id: 'tokens',
    ...COLUMN_SIZES.tokens,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.tokens} />,
    cell: (ctx) => <TraceTokensCell trace={ctx.row.original} />,
  },
  {
    id: 'cost',
    ...COLUMN_SIZES.cost,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.cost} />,
    cell: (ctx) => <TraceCostCell trace={ctx.row.original} />,
  },
  {
    id: 'tags',
    ...COLUMN_SIZES.tags,
    header: () => <FormattedMessage {...TRACE_COLUMN_LABELS.tags} />,
    cell: (ctx) => {
      const { intl, onTraceSelected, onFilterByTag } = getTableMeta(ctx);
      const trace = ctx.row.original;
      return (
        <TraceTagsCell
          trace={trace}
          onSelect={onTraceSelected}
          accessibleLabel={openLabel(intl, trace.trace_id, 'tags')}
          onFilterByTag={onFilterByTag}
        />
      );
    },
  },
];

const EMPTY_EXTRA_COLUMNS: TraceTableColumn[] = [];

/**
 * The visible column defs, in canonical order: the standard columns filtered to `visibleColumns`
 * (order stays canonical — a membership set, not a reorderable list), followed by any consumer
 * `extraColumns`. Callers should memoize the result; `extraColumns` defaults to a stable module-scope
 * empty array so an omitted value doesn't churn the memo.
 */
export const getVisibleColumnDefs = (
  visibleColumns: TraceColumnId[],
  extraColumns: TraceTableColumn[] = EMPTY_EXTRA_COLUMNS,
): TraceTableColumn[] => {
  const visible = new Set<string>(visibleColumns);
  return [...STANDARD_COLUMNS.filter((column) => visible.has(column.id)), ...extraColumns];
};

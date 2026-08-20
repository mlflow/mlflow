import type { SortDirection, TraceColumnId } from './types';

/** Page-size options offered in the pagination bar. */
export const PAGE_SIZE_OPTIONS = [25, 50, 100] as const;

/** Default page size. Smallest option — a compact first screen that stays snappy. */
export const DEFAULT_PAGE_SIZE = 25;

/**
 * Every column the traces table can surface, in canonical render order. This is the render-order
 * source of truth (it drives the `TraceColumnId` union and the module-scope `STANDARD_COLUMNS`),
 * so it must list exactly the ids `STANDARD_COLUMNS` defines.
 */
export const TRACE_COLUMN_IDS = [
  'trace_id',
  'start_time',
  'input',
  'output',
  'session',
  'duration',
  'state',
  'tokens',
  'cost',
  'tags',
] as const;

/**
 * Columns the user can sort by. A cursor-paginated search API can only sort server-side, so sorting
 * a column the API doesn't order by would only reorder the current page and mislead across pages —
 * those headers render without a sort affordance. Kept in one place so the table and any consumer
 * order-by builder can't disagree about what's sortable.
 */
export const SORTABLE_TRACE_COLUMNS: readonly TraceColumnId[] = ['start_time', 'duration'];

const SORTABLE_TRACE_COLUMN_SET = new Set<string>(SORTABLE_TRACE_COLUMNS);

/** Narrows an arbitrary id to a server-sortable `TraceColumnId`. Shared so the table and any consumer
 * order-by/URL builder agree on what's sortable (both derive from `SORTABLE_TRACE_COLUMNS`). */
export const isSortableTraceColumn = (id: string): id is TraceColumnId => SORTABLE_TRACE_COLUMN_SET.has(id);

/** Default sort: newest traces first. */
export const DEFAULT_SORT_COLUMN: TraceColumnId = 'start_time';
export const DEFAULT_SORT_DIR: SortDirection = 'desc';

/**
 * Seed pixel widths per column, sized from the previous flex intent (input/output carry the most
 * content, so they get the most room; id/session are medium; time/duration/tokens/cost are compact).
 * `size` is the initial/reset width; `minSize`/`maxSize` bound dragging. Persisted overrides win over
 * `size` at render time.
 */
export interface ColumnSizeSpec {
  size: number;
  minSize: number;
  maxSize: number;
}

export const COLUMN_SIZES: Record<TraceColumnId, ColumnSizeSpec> = {
  start_time: { size: 120, minSize: 90, maxSize: 320 },
  input: { size: 360, minSize: 160, maxSize: 900 },
  output: { size: 360, minSize: 160, maxSize: 900 },
  session: { size: 140, minSize: 100, maxSize: 480 },
  duration: { size: 100, minSize: 80, maxSize: 240 },
  state: { size: 96, minSize: 72, maxSize: 240 },
  trace_id: { size: 160, minSize: 100, maxSize: 480 },
  tokens: { size: 110, minSize: 80, maxSize: 220 },
  cost: { size: 110, minSize: 80, maxSize: 220 },
  tags: { size: 200, minSize: 120, maxSize: 600 },
};

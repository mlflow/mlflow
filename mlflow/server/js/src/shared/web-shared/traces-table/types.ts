import type { ColumnDef } from '@tanstack/react-table';
import type { ReactNode } from 'react';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import type { To } from 'react-router';
import type { PAGE_SIZE_OPTIONS, TRACE_COLUMN_IDS } from './constants';

/** One of the fixed, known trace columns (canonical render order lives in `TRACE_COLUMN_IDS`). */
export type TraceColumnId = (typeof TRACE_COLUMN_IDS)[number];

export type SortDirection = 'asc' | 'desc';

/** A sortable standard column or a dynamic trace-level assessment column. */
export type TraceSortColumnId = TraceColumnId | `assessment:${string}`;

/** A page-size the pagination bar offers. */
export type PageSize = (typeof PAGE_SIZE_OPTIONS)[number];

/** Already-matched input rows. The consumer owns joining, filtering, and pagination. */
export interface TraceComparisonRow {
  kind?: 'trace';
  id: string;
  input: string;
  runs: {
    runUuid: string;
    label: string;
    color: string;
    trace?: ModelTraceInfoV3;
  }[];
}

/** Already-matched conversations. Turns belong to a run; they are not matched by position. */
export interface SessionComparisonRow {
  kind: 'session';
  id: string;
  runs: {
    runUuid: string;
    label: string;
    color: string;
    sessionId?: string;
    traces: ModelTraceInfoV3[];
  }[];
}

export type TracesComparisonRow = TraceComparisonRow | SessionComparisonRow;

/** Renders one run slot's value with access to its matched row; slot order carries no shared meaning. */
export type ComparisonCellRenderer = (params: {
  trace: ModelTraceInfoV3;
  row: TraceComparisonRow;
  runIndex: number;
}) => ReactNode;

/** Renders a product-owned summary for an extra column in a grouped session header row. */
export type SessionCellRenderer = (traces: ModelTraceInfoV3[]) => ReactNode;

/** A table column definition over a trace row — the type product code passes as an `extraColumns` entry. */
export type TraceTableColumn = ColumnDef<ModelTraceInfoV3> & {
  renderSessionCell?: SessionCellRenderer;
  /** Replaces `cell` in comparison rows only; every other row kind still renders `cell`. */
  renderComparisonCell?: ComparisonCellRenderer;
  /** Additional product options rendered in the column menu, above Hide column. */
  headerMenuContent?: ReactNode;
  /** Plain-text column name for the menu trigger's a11y label when the header is JSX instead of a string. */
  labelText?: string;
  /** `'rich'` lets a consumer render a block header (such as a score summary) instead of a one-line label. */
  headerVariant?: 'standard' | 'rich';
};

/**
 * Resolves the destination the Session cell links to, or `undefined` to render the session as plain
 * text (no link). The only product-specific coupling in the presentational layer: the shared cell
 * keeps its Tag + truncation + `stopPropagation` styling and asks the consumer only for the route.
 */
export type SessionHrefGetter = (params: { trace: ModelTraceInfoV3; sessionId: string }) => To | undefined;

/** Handles activation (click) of a grouped session summary row. */
export type SessionSelectionHandler = (params: { trace: ModelTraceInfoV3; sessionId: string }) => void;

/** Resolves the destination used by trace identity and preview cells. */
export type TraceHrefGetter = (trace: ModelTraceInfoV3) => To | undefined;

export interface TraceColumnHeaderAction {
  label: ReactNode;
  icon: ReactNode;
  onClick: () => void;
}

/** Consumer-computed summary rendered beneath a column's title. */
export interface TraceColumnHeaderSummary {
  content: ReactNode;
  minWidth?: number;
}

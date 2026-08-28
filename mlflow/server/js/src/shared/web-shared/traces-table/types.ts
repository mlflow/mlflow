import type { ColumnDef } from '@tanstack/react-table';
import type { ReactNode } from 'react';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import type { To } from 'react-router';
import type { PAGE_SIZE_OPTIONS, TRACE_COLUMN_IDS } from './constants';

/** One of the fixed, known trace columns (canonical render order lives in `TRACE_COLUMN_IDS`). */
export type TraceColumnId = (typeof TRACE_COLUMN_IDS)[number];

export type SortDirection = 'asc' | 'desc';

/** A page-size the pagination bar offers. */
export type PageSize = (typeof PAGE_SIZE_OPTIONS)[number];

/** Renders a product-owned summary for an extra column in a grouped session header row. */
export type SessionCellRenderer = (traces: ModelTraceInfoV3[]) => ReactNode;

/** A table column definition over a trace row — the type product code passes as an `extraColumns` entry. */
export type TraceTableColumn = ColumnDef<ModelTraceInfoV3> & {
  renderSessionCell?: SessionCellRenderer;
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

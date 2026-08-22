import type { ColumnDef } from '@tanstack/react-table';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import type { To } from 'react-router';
import type { PAGE_SIZE_OPTIONS, TRACE_COLUMN_IDS } from './constants';

/** One of the fixed, known trace columns (canonical render order lives in `TRACE_COLUMN_IDS`). */
export type TraceColumnId = (typeof TRACE_COLUMN_IDS)[number];

export type SortDirection = 'asc' | 'desc';

/** A page-size the pagination bar offers. */
export type PageSize = (typeof PAGE_SIZE_OPTIONS)[number];

/** A table column definition over a trace row — the type product code passes as an `extraColumns` entry. */
export type TraceTableColumn = ColumnDef<ModelTraceInfoV3>;

/**
 * Resolves the destination the Session cell links to, or `undefined` to render the session as plain
 * text (no link). The only product-specific coupling in the presentational layer: the shared cell
 * keeps its Tag + truncation + `stopPropagation` styling and asks the consumer only for the route.
 */
export type SessionHrefGetter = (params: { trace: ModelTraceInfoV3; sessionId: string }) => To | undefined;

/** Resolves the destination used by trace identity and preview cells. */
export type TraceHrefGetter = (trace: ModelTraceInfoV3) => To | undefined;

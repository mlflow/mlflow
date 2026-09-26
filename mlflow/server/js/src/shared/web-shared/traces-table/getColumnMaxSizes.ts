import type { IntlShape } from '@databricks/i18n';
import { getTimeAgoStrings } from '../browse/TimeAgo';
import { formatCostUSD } from '../model-trace-explorer/CostUtils';
import { getTraceCost, getTraceTokenUsage } from '../model-trace-explorer/ModelTraceExplorer.utils';
import { SESSION_ID_METADATA_KEY } from '../model-trace-explorer/constants';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import { COLUMN_SIZES } from './constants';
import { getTraceColumnLabel } from './columnLabels';
import { formatTraceDuration } from './formatTraceDuration';
import type { TraceColumnId } from './types';
import { getTraceStateLabel } from './TraceCell';

const CELL_PADDING_PX = 32;
const TAG_CHROME_PX = 24;
const ICON_PX = 20;

// Managed measures real glyph widths via the datagrid's `getTableStringWidth`; that util isn't
// available in OSS, so we approximate with a fixed per-character width. A slight over-estimate is
// preferable here — it only affects the resize *ceiling*, so erring wide keeps the longest value
// revealable without introducing large stretches of empty space.
const PX_PER_CHAR = 10;
const widthFor = (text: string, chrome = 0) => Array.from(text).length * PX_PER_CHAR + CELL_PADDING_PX + chrome;

const widest = (values: string[]) => Math.max(...values.map((value) => widthFor(value)));

/** Content-derived ceiling for a text-like column, never narrower than its seeded width. */
export const getTextColumnMaxSize = (values: string[], defaultSize: number, chrome = 0): number =>
  Math.ceil(Math.max(defaultSize, widest(values) + chrome));

const validTimeLabel = (trace: ModelTraceInfoV3, intl: IntlShape): string => {
  if (!trace.request_time) {
    return '-';
  }
  const date = new Date(trace.request_time);
  return Number.isNaN(date.getTime()) ? '-' : getTimeAgoStrings({ date, intl }).displayText;
};

/**
 * Content-derived resize ceilings for columns whose rendered values are intrinsically compact.
 * The longest value on the current page remains fully revealable, while a user cannot drag the
 * column hundreds of pixels beyond it. Input/output and tags are intentionally excluded: their
 * useful content grows with available space.
 */
export type ContentColumnMaxSizes = Partial<Record<TraceColumnId, number>>;

export const getContentColumnMaxSizes = (traces: ModelTraceInfoV3[], intl: IntlShape): ContentColumnMaxSizes => {
  const values = {
    trace_id: traces.map((trace) => trace.trace_id || '-'),
    start_time: traces.map((trace) => validTimeLabel(trace, intl)),
    session: traces.map((trace) => trace.trace_metadata?.[SESSION_ID_METADATA_KEY] || '-'),
    duration: traces.map((trace) =>
      trace.execution_duration ? (formatTraceDuration(trace.execution_duration) ?? trace.execution_duration) : '-',
    ),
    state: traces.map((trace) => getTraceStateLabel(trace.state, intl)),
    tokens: traces.map((trace) => String(getTraceTokenUsage(trace)?.total_tokens || '-')),
    cost: traces.map((trace) => {
      const total = getTraceCost(trace)?.total_cost;
      return total === undefined || total === null ? '-' : formatCostUSD(total);
    }),
  };

  const sized = (id: keyof typeof values, chrome = 0) =>
    getTextColumnMaxSize([getTraceColumnLabel(id, intl), ...values[id]], COLUMN_SIZES[id].size, chrome);

  return {
    trace_id: sized('trace_id'),
    start_time: sized('start_time'),
    session: sized('session', TAG_CHROME_PX),
    duration: sized('duration', TAG_CHROME_PX + ICON_PX),
    state: sized('state', TAG_CHROME_PX + ICON_PX),
    tokens: sized('tokens', TAG_CHROME_PX),
    cost: sized('cost', TAG_CHROME_PX),
  };
};

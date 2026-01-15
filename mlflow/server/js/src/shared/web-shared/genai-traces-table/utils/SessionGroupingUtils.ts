import { compact } from 'lodash';

import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import type { EvalTraceComparisonEntry } from '../types';

export type GroupedTraceTableRowData =
  | { type: 'trace'; data: EvalTraceComparisonEntry }
  | { type: 'sessionHeader'; sessionId: string; traces: ModelTraceInfoV3[] };

/**
 * Extract session ID from a ModelTraceInfoV3.
 */
const getSessionIdFromTrace = (trace: ModelTraceInfoV3): string | null => {
  return trace.trace_metadata?.[SESSION_ID_METADATA_KEY] ?? null;
};

/**
 * Collect all ModelTraceInfoV3 objects from a list of entries (currentRunValue only).
 */
const collectTracesFromEntries = (entries: EvalTraceComparisonEntry[]): ModelTraceInfoV3[] => {
  const traces: ModelTraceInfoV3[] = [];

  // for now, only checks the current run as the compare flow
  // will soon be refactored entirely to a different component.
  entries.forEach((entry) => {
    if (entry.currentRunValue?.traceInfo) {
      traces.push(entry.currentRunValue.traceInfo);
    }
  });

  return compact(traces);
};

/**
 * Groups traces by session for display in a table.
 * Returns a flattened array where each session is represented by:
 * 1. A sessionHeader row containing all traces in that session
 * 2. Individual trace rows for each trace in the session
 *
 * Traces without a session ID are appended at the end as standalone trace rows.
 */
export const groupTracesBySessionForTable = (
  currentEvaluationResults: EvalTraceComparisonEntry[],
): GroupedTraceTableRowData[] => {
  const sessionMap: Record<string, EvalTraceComparisonEntry[]> = {};
  const standaloneEntries: EvalTraceComparisonEntry[] = [];

  // Group entries by session ID
  currentEvaluationResults.forEach((entry) => {
    const traceInfo = entry.currentRunValue?.traceInfo;
    if (!traceInfo) {
      return;
    }

    const sessionId = getSessionIdFromTrace(traceInfo);
    if (sessionId) {
      const sessionEntries = sessionMap[sessionId] ?? [];
      sessionMap[sessionId] = [...sessionEntries, entry];
    } else {
      standaloneEntries.push(entry);
    }
  });

  const result: GroupedTraceTableRowData[] = [];

  // Process each session: add header followed by trace rows
  Object.entries(sessionMap).forEach(([sessionId, sessionEntries]) => {
    if (sessionEntries.length === 0) {
      return;
    }

    // Collect all traces for the session header
    const traces = collectTracesFromEntries(sessionEntries);

    // Add session header
    result.push({
      type: 'sessionHeader',
      sessionId,
      traces,
    });

    // Add individual trace rows
    sessionEntries.forEach((entry) => {
      result.push({
        type: 'trace',
        data: entry,
      });
    });
  });

  // Add standalone traces at the end
  standaloneEntries.forEach((entry) => {
    result.push({
      type: 'trace',
      data: entry,
    });
  });

  return result;
};

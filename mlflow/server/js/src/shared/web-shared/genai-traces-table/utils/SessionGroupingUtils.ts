import { compact } from 'lodash';

import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import type { EvalTraceComparisonEntry } from '../types';
import { shouldEnableSessionGrouping } from './FeatureUtils';

export interface SessionHeaderRowData {
  type: 'sessionHeader';
  sessionId: string;
  otherSessionId?: string;
  traces: ModelTraceInfoV3[];
  otherTraces?: ModelTraceInfoV3[];
  goal?: string;
  persona?: string;
}

export type GroupedTraceTableRowData = { type: 'trace'; data: EvalTraceComparisonEntry } | SessionHeaderRowData;

const SIMULATION_GOAL_KEY = 'mlflow.simulation.goal';
const SIMULATION_PERSONA_KEY = 'mlflow.simulation.persona';

/**
 * Extract session ID from a ModelTraceInfoV3.
 */
const getSessionIdFromTrace = (trace: ModelTraceInfoV3): string | null => {
  return trace.trace_metadata?.[SESSION_ID_METADATA_KEY] ?? null;
};

/**
 * Get a matching key for sessions. Sessions are matched by goal and persona metadata
 * when in comparison mode (and session grouping is enabled), falling back to session ID
 * otherwise. This allows sessions with different IDs but the same goal/persona to be
 * grouped together during comparison.
 *
 * @param trace - The trace to get the match key for
 * @param isComparing - Whether we're in comparison mode (goal/persona matching only applies here)
 */
const getSessionMatchKey = (trace: ModelTraceInfoV3, isComparing?: boolean): string => {
  // Only use goal/persona matching when in comparison mode AND session grouping is enabled
  if (isComparing && shouldEnableSessionGrouping()) {
    const goal = trace.trace_metadata?.[SIMULATION_GOAL_KEY];
    const persona = trace.trace_metadata?.[SIMULATION_PERSONA_KEY];

    // Prefer matching by goal/persona since session IDs are often unique per run
    if (goal && persona) {
      return `metadata:${goal}:${persona}`;
    }
  }

  // Fallback to session ID if no goal/persona available, not comparing, or feature is disabled
  const sessionId = getSessionIdFromTrace(trace);
  if (sessionId) {
    return `session:${sessionId}`;
  }

  return `unmatched:${trace.trace_id}`;
};

/**
 * Collect all ModelTraceInfoV3 objects from a list of entries (currentRunValue only).
 */
const collectTracesFromEntries = (entries: EvalTraceComparisonEntry[]): ModelTraceInfoV3[] => {
  const traces: ModelTraceInfoV3[] = [];

  entries.forEach((entry) => {
    if (entry.currentRunValue?.traceInfo) {
      traces.push(entry.currentRunValue.traceInfo);
    }
  });

  return compact(traces);
};

/**
 * Collect all ModelTraceInfoV3 objects from otherRunValue in entries.
 */
const collectOtherTracesFromEntries = (entries: EvalTraceComparisonEntry[]): ModelTraceInfoV3[] => {
  const traces: ModelTraceInfoV3[] = [];

  entries.forEach((entry) => {
    if (entry.otherRunValue?.traceInfo) {
      traces.push(entry.otherRunValue.traceInfo);
    }
  });

  return compact(traces);
};

interface SessionData {
  sessionId: string;
  otherSessionId?: string;
  matchKey: string;
  currentTraces: ModelTraceInfoV3[];
  otherTraces: ModelTraceInfoV3[];
  goal?: string;
  persona?: string;
}

/**
 * Groups traces by session for display in a table.
 * Returns a flattened array where each session is represented by:
 * 1. A sessionHeader row containing all traces in that session
 * 2. Individual trace rows for each trace in the session (only if session is expanded)
 *
 * When comparing runs, sessions are matched by session ID or by goal/persona metadata.
 * Traces without a session ID are appended at the end as standalone trace rows.
 *
 * @param currentEvaluationResults - The evaluation entries to group
 * @param expandedSessions - Set of session IDs that are expanded (show trace rows)
 * @param isComparing - Whether we're in comparison mode
 */
export const groupTracesBySessionForTable = (
  currentEvaluationResults: EvalTraceComparisonEntry[],
  expandedSessions: Set<string>,
  isComparing?: boolean,
): { groupedRows: GroupedTraceTableRowData[]; traceIdToTurnMap: Record<string, number> } => {
  const sessionDataMap: Record<string, SessionData> = {};
  const standaloneEntries: EvalTraceComparisonEntry[] = [];
  const processedOtherTraceIds = new Set<string>();

  // First pass: Group all current run traces by their match key
  currentEvaluationResults.forEach((entry) => {
    const traceInfo = entry.currentRunValue?.traceInfo;
    if (!traceInfo) {
      return;
    }

    const matchKey = getSessionMatchKey(traceInfo, isComparing);
    const sessionId = getSessionIdFromTrace(traceInfo);

    // Handle traces without session ID or goal/persona as standalone
    if (!sessionId && !matchKey.startsWith('metadata:')) {
      standaloneEntries.push(entry);
      return;
    }

    if (!sessionDataMap[matchKey]) {
      const goal = traceInfo.trace_metadata?.[SIMULATION_GOAL_KEY];
      const persona = traceInfo.trace_metadata?.[SIMULATION_PERSONA_KEY];
      sessionDataMap[matchKey] = {
        sessionId: sessionId || matchKey,
        matchKey,
        currentTraces: [],
        otherTraces: [],
        goal,
        persona,
      };
    }
    sessionDataMap[matchKey].currentTraces.push(traceInfo);
  });

  // Second pass: Group all other run traces by their match key and match to existing sessions
  if (isComparing) {
    currentEvaluationResults.forEach((entry) => {
      const otherTraceInfo = entry.otherRunValue?.traceInfo;
      if (!otherTraceInfo) {
        return;
      }

      const traceId = otherTraceInfo.trace_id;
      if (processedOtherTraceIds.has(traceId)) {
        return;
      }
      processedOtherTraceIds.add(traceId);

      const matchKey = getSessionMatchKey(otherTraceInfo, isComparing);
      const sessionId = getSessionIdFromTrace(otherTraceInfo);

      // Handle traces without session ID or goal/persona as standalone
      if (!sessionId && !matchKey.startsWith('metadata:')) {
        // Only add to standalone if not already added via currentRunValue
        if (!entry.currentRunValue?.traceInfo) {
          standaloneEntries.push(entry);
        }
        return;
      }

      // Check if this session already exists (matched by key)
      if (sessionDataMap[matchKey]) {
        // Add to existing session and track the other session ID if different
        sessionDataMap[matchKey].otherTraces.push(otherTraceInfo);
        if (sessionId && sessionId !== sessionDataMap[matchKey].sessionId) {
          sessionDataMap[matchKey].otherSessionId = sessionId;
        }
      } else {
        // Create new session for other run traces only
        const goal = otherTraceInfo.trace_metadata?.[SIMULATION_GOAL_KEY];
        const persona = otherTraceInfo.trace_metadata?.[SIMULATION_PERSONA_KEY];
        sessionDataMap[matchKey] = {
          sessionId: sessionId || matchKey,
          matchKey,
          currentTraces: [],
          otherTraces: [otherTraceInfo],
          goal,
          persona,
        };
      }
    });
  }

  const result: GroupedTraceTableRowData[] = [];
  const traceIdToTurnMap: Record<string, number> = {};

  // Process each session: add header followed by trace rows (if expanded)
  Object.values(sessionDataMap).forEach((sessionData) => {
    const { sessionId, otherSessionId, currentTraces, otherTraces, goal, persona } = sessionData;

    if (currentTraces.length === 0 && otherTraces.length === 0) {
      return;
    }

    // Sort traces by request time ascending within the session (first turn to last turn)
    const sortedCurrentTraces = currentTraces.toSorted((a, b) => {
      const aTime = a.request_time ?? '';
      const bTime = b.request_time ?? '';
      return aTime.localeCompare(bTime);
    });

    const sortedOtherTraces = otherTraces.toSorted((a, b) => {
      const aTime = a.request_time ?? '';
      const bTime = b.request_time ?? '';
      return aTime.localeCompare(bTime);
    });

    // Add session header
    result.push({
      type: 'sessionHeader',
      sessionId,
      otherSessionId,
      traces: sortedCurrentTraces,
      otherTraces: isComparing ? sortedOtherTraces : undefined,
      goal,
      persona,
    });

    // Add individual trace rows only if session is expanded and not comparing
    // In comparison mode with session grouping, sessions are not expandable
    if (expandedSessions.has(sessionId) && !isComparing) {
      // Create a map of trace_id -> sorted index for ordering
      const traceIdToSortIndex = new Map(sortedCurrentTraces.map((t, idx) => [t.trace_id, idx]));

      // Find the original entries for these traces and sort them to match sortedCurrentTraces order
      const entriesForSession = currentEvaluationResults
        .filter((entry) => {
          const traceId = entry.currentRunValue?.traceInfo?.trace_id;
          return traceId && traceIdToSortIndex.has(traceId);
        })
        .toSorted((a, b) => {
          const aIdx = traceIdToSortIndex.get(a.currentRunValue?.traceInfo?.trace_id ?? '') ?? 0;
          const bIdx = traceIdToSortIndex.get(b.currentRunValue?.traceInfo?.trace_id ?? '') ?? 0;
          return aIdx - bIdx;
        });

      entriesForSession.forEach((entry, index) => {
        result.push({
          type: 'trace',
          data: entry,
        });

        const traceId = entry.currentRunValue?.traceInfo?.trace_id;
        if (traceId) {
          traceIdToTurnMap[traceId] = index + 1;
        }
      });
    }
  });

  // Add standalone traces at the end
  standaloneEntries.forEach((entry) => {
    result.push({
      type: 'trace',
      data: entry,
    });
  });

  return { groupedRows: result, traceIdToTurnMap };
};

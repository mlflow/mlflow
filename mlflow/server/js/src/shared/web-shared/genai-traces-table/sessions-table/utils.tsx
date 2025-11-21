import { compact, isNil, sortBy } from 'lodash';

import {
  getTotalTokens,
  type ModelTraceInfoV3,
  SESSION_ID_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';

import type { SessionTableRow } from './types';
import MlflowUtils from '../utils/MlflowUtils';

export const groupTracesBySession = (traces: ModelTraceInfoV3[]) => {
  const sessionIdMap: Record<string, ModelTraceInfoV3[]> = {};

  traces.forEach((trace) => {
    const sessionId = trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
    if (!sessionId) {
      return;
    }

    const sessionTraces = sessionIdMap[sessionId] ?? [];
    sessionIdMap[sessionId] = [...sessionTraces, trace];
  });

  return sessionIdMap;
};

export const getSessionTableRows = (experimentId: string, traces: ModelTraceInfoV3[]): SessionTableRow[] => {
  const sessionIdMap = groupTracesBySession(traces);

  return compact(
    Object.entries(sessionIdMap).map(([sessionId, traces]) => {
      if (traces.length === 0) {
        return null;
      }

      // sort traces within a session by time (earliest first)
      const sortedTraces = sortBy(traces, (trace) => new Date(trace.request_time));
      const firstTrace = sortedTraces[0];

      // take request preview from the first trace
      const requestPreview = firstTrace.request_preview;

      const totalTokens = sortedTraces.reduce((acc, trace) => acc + (getTotalTokens(trace) ?? 0), 0);

      return {
        sessionId,
        requestPreview,
        firstTrace,
        experimentId,
        sessionStartTime: MlflowUtils.formatTimestamp(new Date(firstTrace.request_time)),
        sessionDuration: calculateSessionDuration(traces),
        tokens: totalTokens,
        turns: sortedTraces.length,
      };
    }),
  );
};

const calculateSessionDuration = (traces: ModelTraceInfoV3[]) => {
  const durations = traces.map((trace) => trace.execution_duration);

  if (durations.some((duration) => isNil(duration))) {
    return null;
  }

  const parsedSeconds = durations.map((duration) => parseFloat(duration ?? '0'));
  if (parsedSeconds.some((duration) => isNaN(duration))) {
    return null;
  }

  const totalMs = parsedSeconds.reduce((a, b) => a + b, 0) * 1000;
  return MlflowUtils.formatDuration(totalMs);
};

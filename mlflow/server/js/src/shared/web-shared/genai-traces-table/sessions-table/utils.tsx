import { compact, sortBy } from 'lodash';

import { type ModelTraceInfoV3, SESSION_ID_METADATA_KEY } from '@databricks/web-shared/model-trace-explorer';

export type SessionTableRow = {
  sessionId: string;
  requestPreview?: string;
  firstTrace: ModelTraceInfoV3;
  experimentId: string;
};

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

      return {
        sessionId,
        requestPreview,
        firstTrace,
        experimentId,
      };
    }),
  );
};

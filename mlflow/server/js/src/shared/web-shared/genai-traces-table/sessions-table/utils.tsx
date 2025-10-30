import { compact, sortBy } from 'lodash';

import {
  type ModelTraceInfoV3,
  SESSION_ID_METADATA_KEY,
  SOURCE_NAME_METADATA_KEY,
  SOURCE_TYPE_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';

export type SessionTableRow = {
  sessionId: string;
  requestPreview?: string;
  source?: {
    name: string;
    type: string;
  };
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

      // take request preview from the first trace
      const requestPreview = sortedTraces[0].request_preview;
      const sourceName = sortedTraces[0].trace_metadata?.[SOURCE_NAME_METADATA_KEY];
      const sourceType = sortedTraces[0].trace_metadata?.[SOURCE_TYPE_METADATA_KEY];
      const source =
        sourceName && sourceType
          ? {
              name: sourceName,
              type: sourceType,
            }
          : undefined;

      return {
        sessionId,
        requestPreview,
        source,
        experimentId,
      };
    }),
  );
};

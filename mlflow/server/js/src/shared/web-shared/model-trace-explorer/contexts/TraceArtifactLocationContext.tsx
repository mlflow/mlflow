import React, { createContext, useContext, useMemo, type ReactNode } from 'react';

import type { ModelTrace } from '../ModelTrace.types';
import { getModelTraceId } from '../ModelTraceExplorer.utils';
import { getTraceArtifactLocation } from '../attachment-utils';

interface TraceArtifactLocationContextValue {
  traceId?: string;
  artifactLocation?: string;
}

const TraceArtifactLocationContext = createContext<TraceArtifactLocationContextValue>({});

/**
 * Publishes the displayed trace's `mlflow.artifactLocation` so attachment fetches can read
 * directly from an MLflow artifact-proxy URI instead of routing through the tracking server.
 */
export const TraceArtifactLocationContextProvider = ({
  children,
  modelTrace,
}: {
  children: ReactNode;
  modelTrace: ModelTrace;
}): JSX.Element => {
  const value = useMemo(
    () => ({
      traceId: getModelTraceId(modelTrace),
      artifactLocation: getTraceArtifactLocation(modelTrace.info),
    }),
    [modelTrace],
  );

  return <TraceArtifactLocationContext.Provider value={value}>{children}</TraceArtifactLocationContext.Provider>;
};

/**
 * Returns the stored artifact location for `traceId`, or undefined when the surrounding
 * provider describes a different trace (or there is no provider at all). Attachments for an
 * unknown trace must keep going through the tracking server.
 */
export const useTraceArtifactLocation = (traceId?: string): string | undefined => {
  const { traceId: contextTraceId, artifactLocation } = useContext(TraceArtifactLocationContext);
  return traceId && contextTraceId === traceId ? artifactLocation : undefined;
};

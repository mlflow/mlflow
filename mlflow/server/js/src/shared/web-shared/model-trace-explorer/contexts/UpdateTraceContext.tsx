import React, { useMemo } from 'react';

import type { ModelTrace } from '../ModelTrace.types';

const ModelTraceExplorerUpdateTraceContext = React.createContext<{
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
}>({
  sqlWarehouseId: undefined,
  modelTraceInfo: undefined,
  invalidateTraceQuery: undefined,
  chatSessionId: undefined,
});

/**
 * Provides configuration context used to update trace data (assessments, tags).
 * Contains:
 * - an ID of the SQL warehouse to use for queries
 * - info of the currently selected model trace
 */
export const ModelTraceExplorerUpdateTraceContextProvider = ({
  sqlWarehouseId,
  modelTraceInfo,
  children,
  invalidateTraceQuery,
  chatSessionId,
}: {
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  children: React.ReactNode;
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
}) => {
  const contextValue = useMemo(
    () => ({ sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId }),
    [sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId],
  );
  return (
    <ModelTraceExplorerUpdateTraceContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerUpdateTraceContext.Provider>
  );
};

export const useModelTraceExplorerUpdateTraceContext = () => React.useContext(ModelTraceExplorerUpdateTraceContext);

import React, { useMemo } from 'react';

import type { ModelTrace } from '../ModelTrace.types';

const ModelTraceExplorerUpdateTraceContext = React.createContext<{
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  invalidateTraceQuery?: (traceId?: string) => void;
}>({
  sqlWarehouseId: undefined,
  modelTraceInfo: undefined,
  invalidateTraceQuery: undefined,
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
}: {
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  children: React.ReactNode;
  invalidateTraceQuery?: (traceId?: string) => void;
}) => {
  const contextValue = useMemo(
    () => ({ sqlWarehouseId, modelTraceInfo, invalidateTraceQuery }),
    [sqlWarehouseId, modelTraceInfo, invalidateTraceQuery],
  );
  return (
    <ModelTraceExplorerUpdateTraceContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerUpdateTraceContext.Provider>
  );
};

export const useModelTraceExplorerUpdateTraceContext = () => React.useContext(ModelTraceExplorerUpdateTraceContext);

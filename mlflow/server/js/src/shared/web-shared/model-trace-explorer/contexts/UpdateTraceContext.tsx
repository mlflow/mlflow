import React, { useMemo } from 'react';

import type { ModelTrace } from '../ModelTrace.types';

const ModelTraceExplorerUpdateTraceContext = React.createContext<{
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
}>({
  sqlWarehouseId: undefined,
  modelTraceInfo: undefined,
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
}: {
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  children: React.ReactNode;
}) => {
  const contextValue = useMemo(() => ({ sqlWarehouseId, modelTraceInfo }), [sqlWarehouseId, modelTraceInfo]);
  return (
    <ModelTraceExplorerUpdateTraceContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerUpdateTraceContext.Provider>
  );
};

export const useModelTraceExplorerUpdateTraceContext = () => React.useContext(ModelTraceExplorerUpdateTraceContext);

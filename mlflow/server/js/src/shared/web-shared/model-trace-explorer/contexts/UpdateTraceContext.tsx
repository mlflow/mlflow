import React, { useMemo } from 'react';

import type { ModelTrace } from '../ModelTrace.types';

const ModelTraceExplorerUpdateTraceContext = React.createContext<{
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
  /**
   * Opt-in "edit tags" affordance. When set, the header's Tags section renders an Edit/Add-tags
   * button that invokes this with the currently-shown trace info. Undefined for consumers that don't
   * support tag editing (e.g. v3), so their header is unchanged.
   */
  onEditTags?: (traceInfo: ModelTrace['info']) => void;
}>({
  sqlWarehouseId: undefined,
  modelTraceInfo: undefined,
  invalidateTraceQuery: undefined,
  chatSessionId: undefined,
  onEditTags: undefined,
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
  onEditTags,
}: {
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  children: React.ReactNode;
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
  onEditTags?: (traceInfo: ModelTrace['info']) => void;
}) => {
  const contextValue = useMemo(
    () => ({ sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId, onEditTags }),
    [sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId, onEditTags],
  );
  return (
    <ModelTraceExplorerUpdateTraceContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerUpdateTraceContext.Provider>
  );
};

export const useModelTraceExplorerUpdateTraceContext = () => React.useContext(ModelTraceExplorerUpdateTraceContext);

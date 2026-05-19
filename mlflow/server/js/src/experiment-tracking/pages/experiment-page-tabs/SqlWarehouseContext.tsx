import React, { createContext, useContext, useMemo, useState } from 'react';
import type { ModelTraceSearchLocation } from '@databricks/web-shared/model-trace-explorer';
import { isV4TraceLocation } from '@databricks/web-shared/genai-traces-table';
import { usePersistedSqlWarehouseId } from './usePersistedSqlWarehouseId';

export type TraceSearchLocations = ModelTraceSearchLocation[];

interface SqlWarehouseContextValue {
  warehouseId: string | undefined | null;
  setWarehouseId: (id: string | undefined | null) => void;
  warehousesLoading: boolean;
  setWarehousesLoading: (loading: boolean) => void;
  traceSearchLocations?: TraceSearchLocations;
  // True when the experiment uses a V4 trace location (UC schema or table prefix).
  hasV4Location?: boolean;
}

const SqlWarehouseContext = createContext<SqlWarehouseContextValue | null>(null);

/**
 * Provider that backs the shared warehouse state with localStorage
 * via usePersistedSqlWarehouseId, so all tabs see the same selection.
 */
export const SqlWarehouseContextProvider = ({
  children,
  experimentId,
  traceSearchLocations = [],
}: {
  children: React.ReactNode;
  experimentId: string;
  traceSearchLocations?: TraceSearchLocations;
}) => {
  const [warehouseId, setWarehouseId] = usePersistedSqlWarehouseId(experimentId);
  const [warehousesLoading, setWarehousesLoading] = useState(false);

  const hasV4Location = traceSearchLocations?.some(isV4TraceLocation);

  const value = useMemo(
    () => ({
      warehouseId,
      setWarehouseId,
      warehousesLoading,
      setWarehousesLoading,
      traceSearchLocations,
      hasV4Location,
    }),
    [warehouseId, setWarehouseId, warehousesLoading, setWarehousesLoading, traceSearchLocations, hasV4Location],
  );

  return <SqlWarehouseContext.Provider value={value}>{children}</SqlWarehouseContext.Provider>;
};

/**
 * Consume the shared warehouse context. Must be used inside SqlWarehouseContextProvider.
 */
export const useSqlWarehouseContext = (): SqlWarehouseContextValue => {
  const context = useContext(SqlWarehouseContext);
  if (!context) {
    throw new Error('useSqlWarehouseContext must be used within a SqlWarehouseContextProvider');
  }
  return context;
};

/**
 * Safe variant that returns null when no provider is present (e.g. in OSS).
 */
export const useSqlWarehouseContextSafe = (): SqlWarehouseContextValue | null => {
  return useContext(SqlWarehouseContext);
};

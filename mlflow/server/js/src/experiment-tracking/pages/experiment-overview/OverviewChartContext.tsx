import React, { createContext, useContext, useMemo } from 'react';

/**
 * Context value for overview chart props shared across all chart components.
 * This eliminates prop drilling through intermediate components.
 */
export interface OverviewChartContextValue {
  experimentId: string;
  startTimeMs?: number;
  endTimeMs?: number;
  /** Time interval in seconds for grouping metrics by time bucket */
  timeIntervalSeconds: number;
  /** Pre-computed array of timestamp (ms) for all time buckets in the range */
  timeBuckets: number[];
  /** Optional filter expressions to apply to all chart queries (e.g. `trace.tag.\`mlflow.gateway.provider\` = "openai"`) */
  filters?: string[];
}

const OverviewChartContext = createContext<OverviewChartContextValue | null>(null);

interface OverviewChartProviderProps extends OverviewChartContextValue {
  children: React.ReactNode;
}

/**
 * Provider component that supplies chart props to all descendant components.
 * Wrap chart sections with this provider to enable context-based prop access.
 */
export const OverviewChartProvider: React.FC<OverviewChartProviderProps> = ({
  children,
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
  filters,
}) => {
  const value = useMemo(
    () => ({
      experimentId,
      startTimeMs,
      endTimeMs,
      timeIntervalSeconds,
      timeBuckets,
      filters,
    }),
    [experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters],
  );

  return <OverviewChartContext.Provider value={value}>{children}</OverviewChartContext.Provider>;
};

/**
 * Hook to access overview chart props from context.
 * Must be used within an OverviewChartProvider.
 */
export function useOverviewChartContext(): OverviewChartContextValue {
  const context = useContext(OverviewChartContext);
  if (!context) {
    throw new Error('useOverviewChartContext must be used within an OverviewChartProvider');
  }
  return context;
}

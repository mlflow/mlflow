import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceTokenStatsChart = React.lazy(() =>
  import('./TraceTokenStatsChart').then((module) => ({ default: module.TraceTokenStatsChart })),
);

interface LazyTraceTokenStatsChartProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
}

export const LazyTraceTokenStatsChart: React.FC<LazyTraceTokenStatsChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceTokenStatsChart {...props} />
  </React.Suspense>
);

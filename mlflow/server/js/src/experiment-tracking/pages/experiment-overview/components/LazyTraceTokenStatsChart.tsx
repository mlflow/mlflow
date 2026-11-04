import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceTokenStatsChart = React.lazy(() =>
  import('./TraceTokenStatsChart').then((module) => ({ default: module.TraceTokenStatsChart })),
);

interface LazyTraceTokenStatsChartProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
}

export const LazyTraceTokenStatsChart: React.FC<LazyTraceTokenStatsChartProps> = (props) => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceTokenStatsChart {...props} />
  </React.Suspense>
);

import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceRequestsChart = React.lazy(() =>
  import('./TraceRequestsChart').then((module) => ({ default: module.TraceRequestsChart })),
);

interface LazyTraceRequestsChartProps {
  title?: React.ReactNode;
}

export const LazyTraceRequestsChart: React.FC<LazyTraceRequestsChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceRequestsChart {...props} />
  </React.Suspense>
);

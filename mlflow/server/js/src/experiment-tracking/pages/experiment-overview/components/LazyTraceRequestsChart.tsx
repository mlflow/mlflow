import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceRequestsChart = React.lazy(() =>
  import('./TraceRequestsChart').then((module) => ({ default: module.TraceRequestsChart })),
);

interface LazyTraceRequestsChartProps {
  title?: React.ReactNode;
}

export const LazyTraceRequestsChart: React.FC<LazyTraceRequestsChartProps> = (props) => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceRequestsChart {...props} />
  </React.Suspense>
);

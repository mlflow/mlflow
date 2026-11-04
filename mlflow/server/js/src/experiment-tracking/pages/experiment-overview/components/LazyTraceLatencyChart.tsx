import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceLatencyChart = React.lazy(() =>
  import('./TraceLatencyChart').then((module) => ({ default: module.TraceLatencyChart })),
);

export const LazyTraceLatencyChart: React.FC = () => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceLatencyChart />
  </React.Suspense>
);

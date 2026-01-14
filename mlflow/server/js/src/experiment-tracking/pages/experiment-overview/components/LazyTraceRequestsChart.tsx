import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceRequestsChart = React.lazy(() =>
  import('./TraceRequestsChart').then((module) => ({ default: module.TraceRequestsChart })),
);

export const LazyTraceRequestsChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceRequestsChart />
  </React.Suspense>
);

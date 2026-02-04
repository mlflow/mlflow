import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceTokenStatsChart = React.lazy(() =>
  import('./TraceTokenStatsChart').then((module) => ({ default: module.TraceTokenStatsChart })),
);

export const LazyTraceTokenStatsChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceTokenStatsChart />
  </React.Suspense>
);

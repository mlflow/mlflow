import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceTokenUsageChart = React.lazy(() =>
  import('./TraceTokenUsageChart').then((module) => ({ default: module.TraceTokenUsageChart })),
);

export const LazyTraceTokenUsageChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceTokenUsageChart />
  </React.Suspense>
);

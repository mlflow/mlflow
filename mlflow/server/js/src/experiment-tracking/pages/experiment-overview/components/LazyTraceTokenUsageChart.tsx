import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceTokenUsageChart = React.lazy(() =>
  import('./TraceTokenUsageChart').then((module) => ({ default: module.TraceTokenUsageChart })),
);

export const LazyTraceTokenUsageChart: React.FC = () => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceTokenUsageChart />
  </React.Suspense>
);

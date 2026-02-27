import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceErrorsChart = React.lazy(() =>
  import('./TraceErrorsChart').then((module) => ({ default: module.TraceErrorsChart })),
);

export const LazyTraceErrorsChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceErrorsChart />
  </React.Suspense>
);

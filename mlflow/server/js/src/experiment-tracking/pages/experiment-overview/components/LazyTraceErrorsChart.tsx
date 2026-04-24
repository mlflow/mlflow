import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceErrorsChart = React.lazy(() =>
  import('./TraceErrorsChart').then((module) => ({ default: module.TraceErrorsChart })),
);

interface LazyTraceErrorsChartProps {
  enableTraceNavigation?: boolean;
}

export const LazyTraceErrorsChart: React.FC<LazyTraceErrorsChartProps> = ({ enableTraceNavigation }) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceErrorsChart enableTraceNavigation={enableTraceNavigation} />
  </React.Suspense>
);

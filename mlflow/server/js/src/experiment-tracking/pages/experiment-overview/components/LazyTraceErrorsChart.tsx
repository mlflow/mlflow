import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceErrorsChart = React.lazy(() =>
  import('./TraceErrorsChart').then((module) => ({ default: module.TraceErrorsChart })),
);

interface LazyTraceErrorsChartProps {
  enableTraceNavigation?: boolean;
}

export const LazyTraceErrorsChart: React.FC<LazyTraceErrorsChartProps> = ({ enableTraceNavigation }) => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceErrorsChart enableTraceNavigation={enableTraceNavigation} />
  </React.Suspense>
);

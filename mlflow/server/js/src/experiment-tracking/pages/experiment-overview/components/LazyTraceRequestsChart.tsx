import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { OverviewChartProps } from '../types';

const TraceRequestsChart = React.lazy(() =>
  import('./TraceRequestsChart').then((module) => ({ default: module.TraceRequestsChart })),
);

export const LazyTraceRequestsChart: React.FC<OverviewChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceRequestsChart {...props} />
  </React.Suspense>
);

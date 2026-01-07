import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { OverviewChartProps } from '../types';

const TraceLatencyChart = React.lazy(() =>
  import('./TraceLatencyChart').then((module) => ({ default: module.TraceLatencyChart })),
);

export const LazyTraceLatencyChart: React.FC<OverviewChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceLatencyChart {...props} />
  </React.Suspense>
);

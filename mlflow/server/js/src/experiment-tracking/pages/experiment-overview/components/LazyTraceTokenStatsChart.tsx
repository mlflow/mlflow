import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { OverviewChartProps } from '../types';

const TraceTokenStatsChart = React.lazy(() =>
  import('./TraceTokenStatsChart').then((module) => ({ default: module.TraceTokenStatsChart })),
);

export const LazyTraceTokenStatsChart: React.FC<OverviewChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceTokenStatsChart {...props} />
  </React.Suspense>
);

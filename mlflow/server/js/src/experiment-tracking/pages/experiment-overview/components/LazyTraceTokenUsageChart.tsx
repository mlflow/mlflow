import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { OverviewChartProps } from '../types';

const TraceTokenUsageChart = React.lazy(() =>
  import('./TraceTokenUsageChart').then((module) => ({ default: module.TraceTokenUsageChart })),
);

export const LazyTraceTokenUsageChart: React.FC<OverviewChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceTokenUsageChart {...props} />
  </React.Suspense>
);

import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

const ToolLatencyChart = React.lazy(() =>
  import('./ToolLatencyChart').then((module) => ({ default: module.ToolLatencyChart })),
);

export const LazyToolLatencyChart: React.FC<OverviewChartProps> = (props) => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolLatencyChart {...props} />
  </Suspense>
);

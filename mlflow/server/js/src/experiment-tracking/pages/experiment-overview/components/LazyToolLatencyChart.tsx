import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';

const ToolLatencyChart = React.lazy(() =>
  import('./ToolLatencyChart').then((module) => ({ default: module.ToolLatencyChart })),
);

export const LazyToolLatencyChart: React.FC = () => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolLatencyChart />
  </Suspense>
);

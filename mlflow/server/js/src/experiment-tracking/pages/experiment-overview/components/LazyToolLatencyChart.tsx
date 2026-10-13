import React from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';

const ToolLatencyChart = React.lazy(() =>
  import('./ToolLatencyChart').then((module) => ({ default: module.ToolLatencyChart })),
);

export const LazyToolLatencyChart: React.FC = () => (
  <React.Suspense fallback={<OverviewChartLoadingState />}>
    <ToolLatencyChart />
  </React.Suspense>
);

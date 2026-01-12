import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';

const ToolUsageChart = React.lazy(() =>
  import('./ToolUsageChart').then((module) => ({ default: module.ToolUsageChart })),
);

export const LazyToolUsageChart: React.FC = () => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolUsageChart />
  </Suspense>
);

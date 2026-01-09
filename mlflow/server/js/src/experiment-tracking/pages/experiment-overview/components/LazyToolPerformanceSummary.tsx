import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';

const ToolPerformanceSummary = React.lazy(() =>
  import('./ToolPerformanceSummary').then((module) => ({ default: module.ToolPerformanceSummary })),
);

export const LazyToolPerformanceSummary: React.FC = () => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolPerformanceSummary />
  </Suspense>
);

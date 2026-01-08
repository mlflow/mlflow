import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

const ToolPerformanceSummary = React.lazy(() =>
  import('./ToolPerformanceSummary').then((module) => ({ default: module.ToolPerformanceSummary })),
);

export const LazyToolPerformanceSummary: React.FC<Omit<OverviewChartProps, 'timeIntervalSeconds' | 'timeBuckets'>> = (
  props,
) => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolPerformanceSummary {...props} />
  </Suspense>
);

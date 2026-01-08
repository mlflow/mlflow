import React, { Suspense } from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

const ToolUsageChart = React.lazy(() =>
  import('./ToolUsageChart').then((module) => ({ default: module.ToolUsageChart })),
);

export const LazyToolUsageChart: React.FC<OverviewChartProps> = (props) => (
  <Suspense fallback={<OverviewChartLoadingState />}>
    <ToolUsageChart {...props} />
  </Suspense>
);

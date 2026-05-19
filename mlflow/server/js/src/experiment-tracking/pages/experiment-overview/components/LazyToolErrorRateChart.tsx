import React from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';
import type { ToolErrorRateChartProps } from './ToolErrorRateChart';

const ToolErrorRateChart = React.lazy(() =>
  import('./ToolErrorRateChart').then((module) => ({ default: module.ToolErrorRateChart })),
);

export const LazyToolErrorRateChart: React.FC<ToolErrorRateChartProps> = (props) => (
  <React.Suspense fallback={<OverviewChartLoadingState />}>
    <ToolErrorRateChart {...props} />
  </React.Suspense>
);

import React, { Suspense } from 'react';
import { ChartLoadingState } from './ChartCardWrapper';
import type { ToolErrorRateChartProps } from './ToolErrorRateChart';

const ToolErrorRateChart = React.lazy(() =>
  import('./ToolErrorRateChart').then((module) => ({ default: module.ToolErrorRateChart })),
);

export const LazyToolErrorRateChart: React.FC<ToolErrorRateChartProps> = (props) => (
  <Suspense fallback={<ChartLoadingState />}>
    <ToolErrorRateChart {...props} />
  </Suspense>
);

import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { OverviewChartProps } from '../types';

const TraceErrorsChart = React.lazy(() =>
  import('./TraceErrorsChart').then((module) => ({ default: module.TraceErrorsChart })),
);

export const LazyTraceErrorsChart: React.FC<OverviewChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceErrorsChart {...props} />
  </React.Suspense>
);

import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { TraceRequestsChartProps } from './TraceRequestsChart';

const TraceRequestsChart = React.lazy(() =>
  import('./TraceRequestsChart').then((module) => ({ default: module.TraceRequestsChart })),
);

export const LazyTraceRequestsChart: React.FC<TraceRequestsChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceRequestsChart {...props} />
  </React.Suspense>
);

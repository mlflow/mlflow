import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceCostOverTimeChart = React.lazy(() =>
  import('./TraceCostOverTimeChart').then((module) => ({ default: module.TraceCostOverTimeChart })),
);

export const LazyTraceCostOverTimeChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceCostOverTimeChart />
  </React.Suspense>
);

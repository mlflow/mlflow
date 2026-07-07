import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';

const TraceCostBreakdownChart = React.lazy(() =>
  import('./TraceCostBreakdownChart').then((module) => ({ default: module.TraceCostBreakdownChart })),
);

export const LazyTraceCostBreakdownChart: React.FC = () => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceCostBreakdownChart />
  </React.Suspense>
);

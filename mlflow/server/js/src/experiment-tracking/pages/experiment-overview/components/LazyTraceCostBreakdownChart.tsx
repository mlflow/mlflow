import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceCostBreakdownChart = React.lazy(() =>
  import('./TraceCostBreakdownChart').then((module) => ({ default: module.TraceCostBreakdownChart })),
);

export const LazyTraceCostBreakdownChart: React.FC = () => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceCostBreakdownChart />
  </React.Suspense>
);

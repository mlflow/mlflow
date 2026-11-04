import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';

const TraceCostOverTimeChart = React.lazy(() =>
  import('./TraceCostOverTimeChart').then((module) => ({ default: module.TraceCostOverTimeChart })),
);

export const LazyTraceCostOverTimeChart: React.FC = () => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceCostOverTimeChart />
  </React.Suspense>
);

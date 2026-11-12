import React from 'react';
import { OverviewChartLoadingState } from './OverviewChartComponents';

const TopUsersTable = React.lazy(() => import('./TopUsersTable').then((module) => ({ default: module.TopUsersTable })));

export const LazyTopUsersTable: React.FC<React.ComponentProps<typeof TopUsersTable>> = (props) => (
  <React.Suspense fallback={<OverviewChartLoadingState />}>
    <TopUsersTable {...props} />
  </React.Suspense>
);

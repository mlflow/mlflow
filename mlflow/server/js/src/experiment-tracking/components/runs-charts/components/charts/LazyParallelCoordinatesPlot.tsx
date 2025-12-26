import { LegacySkeleton } from '@databricks/design-system';
// eslint-disable-next-line no-restricted-imports -- grandfathering, see go/ui-bestpractices
import React, { Suspense } from 'react';

const ParallelCoordinatesPlot = React.lazy(() => import('./ParallelCoordinatesPlot'));

const LazyParallelCoordinatesPlot = ({ fallback, ...props }: any) => {
  return (
    <Suspense fallback={fallback ?? <LegacySkeleton />}>
      <ParallelCoordinatesPlot {...props} />
    </Suspense>
  );
};

export default LazyParallelCoordinatesPlot;

import { LegacySkeleton } from '@databricks/design-system';
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

import { LegacySkeleton } from '@databricks/design-system';
import React from 'react';

const ParallelCoordinatesPlot = React.lazy(() => import('./ParallelCoordinatesPlot'));

const LazyParallelCoordinatesPlot = ({ fallback, ...props }: any) => {
  return (
    <React.Suspense fallback={fallback ?? <LegacySkeleton />}>
      <ParallelCoordinatesPlot {...props} />
    </React.Suspense>
  );
};

export default LazyParallelCoordinatesPlot;

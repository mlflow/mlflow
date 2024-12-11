import { LegacySkeleton } from '@databricks/design-system';
import React, { Suspense } from 'react';

const ParallelCoordinatesPlot = React.lazy(() => import('./ParallelCoordinatesPlot'));

const LazyParallelCoordinatesPlot = (props: any) => {
  return (
    <Suspense fallback={<LegacySkeleton />}>
      <ParallelCoordinatesPlot {...props}></ParallelCoordinatesPlot>
    </Suspense>
  );
};

export default LazyParallelCoordinatesPlot;

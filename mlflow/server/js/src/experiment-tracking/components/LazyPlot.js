import React from 'react';
import { Skeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../common/components/error-boundaries/SectionErrorBoundary';

const Plot = React.lazy(() => import('react-plotly.js'));

export const LazyPlot = (props) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<Skeleton active />}>
      <Plot {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

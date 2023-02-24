import React from 'react';
import { Skeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactMapView = React.lazy(() => import('./ShowArtifactMapView'));

export const LazyShowArtifactMapView = (props) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<Skeleton active />}>
      <ShowArtifactMapView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

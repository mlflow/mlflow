import React from 'react';
import { Skeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactTableView = React.lazy(() => import('./ShowArtifactTableView'));

export const LazyShowArtifactTableView = (props) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<Skeleton active />}>
      <ShowArtifactTableView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

import React from 'react';
import { Skeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactPdfView = React.lazy(() => import('./ShowArtifactPdfView'));

export const LazyShowArtifactPdfView = (props) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<Skeleton active />}>
      <ShowArtifactPdfView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

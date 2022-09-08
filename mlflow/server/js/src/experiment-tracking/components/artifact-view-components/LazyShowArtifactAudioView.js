import React from 'react';
import { Skeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactAudioView = React.lazy(() => import('./ShowArtifactAudioView'));

export const LazyShowArtifactAudioView = (props) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<Skeleton active />}>
      <ShowArtifactAudioView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

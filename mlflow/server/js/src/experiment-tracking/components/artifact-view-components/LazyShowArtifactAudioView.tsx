import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactAudioView = React.lazy(() => import('./ShowArtifactAudioView'));

export const LazyShowArtifactAudioView = (props: any) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<LegacySkeleton active />}>
      <ShowArtifactAudioView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

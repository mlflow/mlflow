import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';
import type { ShowArtifactAudioViewProps } from './ShowArtifactAudioView';

const ShowArtifactAudioView = React.lazy(() => import('./ShowArtifactAudioView'));

export const LazyShowArtifactAudioView = (props: ShowArtifactAudioViewProps) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<LegacySkeleton active />}>
      <ShowArtifactAudioView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

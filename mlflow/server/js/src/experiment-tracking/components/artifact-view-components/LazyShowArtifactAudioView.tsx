import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';
import type { ShowArtifactAudioViewProps } from './ShowArtifactAudioView';

const ShowArtifactAudioView = React.lazy(() => import('./ShowArtifactAudioView'));

export const LazyShowArtifactAudioView = (props: ShowArtifactAudioViewProps) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<ContentSkeleton />}>
      <ShowArtifactAudioView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

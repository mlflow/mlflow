import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactMarkdownView = React.lazy(() => import('./ShowArtifactMarkdownView'));

export const LazyShowArtifactMarkdownView = (props: any) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<LegacySkeleton active />}>
      <ShowArtifactMarkdownView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

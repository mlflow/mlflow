import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactMarkdownView = React.lazy(() => import('./ShowArtifactMarkdownView'));

export const LazyShowArtifactMarkdownView = (props: React.ComponentProps<typeof ShowArtifactMarkdownView>) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<ContentSkeleton />}>
      <ShowArtifactMarkdownView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

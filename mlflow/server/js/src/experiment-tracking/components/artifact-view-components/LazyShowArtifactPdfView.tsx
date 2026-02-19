/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../../common/components/error-boundaries/SectionErrorBoundary';

const ShowArtifactPdfView = React.lazy(() => import('./ShowArtifactPdfView'));

export const LazyShowArtifactPdfView = (props: any) => (
  <SectionErrorBoundary>
    <React.Suspense fallback={<LegacySkeleton active />}>
      <ShowArtifactPdfView {...props} />
    </React.Suspense>
  </SectionErrorBoundary>
);

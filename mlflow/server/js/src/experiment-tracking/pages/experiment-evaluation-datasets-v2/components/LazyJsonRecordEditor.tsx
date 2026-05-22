import { Suspense, lazy } from 'react';
import { GenericSkeleton } from '@databricks/design-system';
import type { JsonRecordEditorProps } from './JsonRecordEditor';

/**
 * Lazy-loads `JsonRecordEditor` so the Monaco bundle (~1 MB gz with our restricted languages
 * config) stays out of the main app chunk. The record side panel is the only consumer in
 * v2, so users who never open it never download Monaco.
 */
const LazyEditor = lazy(async () => {
  const mod = await import('./JsonRecordEditor');
  return { default: mod.JsonRecordEditor };
});

export const LazyJsonRecordEditor = (props: JsonRecordEditorProps) => (
  <Suspense
    fallback={
      <GenericSkeleton
        style={{ height: props.height ?? '240px', width: '100%' }}
        loadingDescription="JsonRecordEditor"
      />
    }
  >
    <LazyEditor {...props} />
  </Suspense>
);

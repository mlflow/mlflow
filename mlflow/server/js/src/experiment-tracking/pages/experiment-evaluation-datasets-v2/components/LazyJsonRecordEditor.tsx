import { lazy } from 'react';
import { Suspense } from '@databricks/web-shared/react';
import { LoadingDescription } from '@databricks/web-shared/metrics';
import { lazilyImportEditorModule } from '@databricks/editor/lazyImport';
import { GenericSkeleton } from '@databricks/design-system';
import type { JsonRecordEditorProps } from './JsonRecordEditor';

// Lazy-load Monaco so the records page doesn't pull ~1MB of editor JS until the user
// actually opens the drawer.
const LazyEditor = lazy(() =>
  lazilyImportEditorModule(
    async () => {
      const mod = await import('./JsonRecordEditor');
      return { default: mod.JsonRecordEditor };
    },
    'NotebookNativeFeaturePlugin',
    { includeLsp: false },
  ),
);

export const LazyJsonRecordEditor = (props: JsonRecordEditorProps) => {
  return (
    <Suspense
      description={LoadingDescription.MONACO_EDITOR_LOADING}
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
};

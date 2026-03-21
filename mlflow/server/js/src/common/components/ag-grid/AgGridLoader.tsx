import type { AgGridReactProps, AgReactUiProps } from '@ag-grid-community/react';
import { Spinner } from '@databricks/design-system';
import React from 'react';

const MLFlowAgGridImpl = React.lazy(() => import('./AgGrid'));

/**
 * A simple loader that will lazily load MLflow's ag grid implementation.
 * Extracted to a separate module for testing purposes.
 */
export const MLFlowAgGridLoader = (props: AgGridReactProps | AgReactUiProps) => (
  <React.Suspense
    fallback={
      <div
        css={(cssTheme) => ({
          display: 'flex',
          justifyContent: 'center',
          margin: cssTheme.spacing.md,
        })}
      >
        <Spinner />
      </div>
    }
  >
    <MLFlowAgGridImpl {...props} />
  </React.Suspense>
);

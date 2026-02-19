import React, { useMemo } from 'react';
import ReactDOM from 'react-dom';

import { IntlProvider } from '@databricks/i18n';
import { SupportsDuBoisThemes } from '@databricks/web-shared/design-system';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { setActiveWorkspace } from '@mlflow/mlflow/src/workspaces/utils/WorkspaceUtils';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';

import './index.css';

// Set workspace from URL query params before rendering so that all API calls
// include the X-MLFLOW-WORKSPACE header.
const workspaceParam = new URLSearchParams(window.location.search).get('workspace');
if (workspaceParam) {
  setActiveWorkspace(workspaceParam);
}

const LazyModelTraceExplorer = React.lazy(() =>
  import('@databricks/web-shared/model-trace-explorer').then((module) => ({
    default: module.ModelTraceExplorerOSSNotebookRenderer,
  })),
);

const getLazyDesignSystem = () => import('@databricks/design-system');

const LazyDesignSystemProvider = React.lazy(() =>
  getLazyDesignSystem().then((module) => ({ default: module.DesignSystemProvider })),
);

const LazyApplyGlobalStyles = React.lazy(() =>
  getLazyDesignSystem().then((module) => ({ default: module.ApplyGlobalStyles })),
);

const LazyDesignSystemContext = React.lazy(() =>
  getLazyDesignSystem().then((module) => ({ default: module.DesignSystemEventProvider })),
);

const DesignSystemProviders: React.FC<React.PropsWithChildren<unknown>> = ({ children }) => {
  return (
    <SupportsDuBoisThemes>
      <LazyDesignSystemContext callback={() => {}}>
        <LazyDesignSystemProvider>
          <LazyApplyGlobalStyles />
          {children}
        </LazyDesignSystemProvider>
      </LazyDesignSystemContext>
    </SupportsDuBoisThemes>
  );
};

const FLAG_OVERRIDES: Record<string, boolean> = {
  // without this, the tags look really ugly in OSS
  'databricks.fe.designsystem.useNewTagColors': true,
  'databricks.fe.traceExplorer.enableSummaryView': true,
};

export const AppComponent = () => {
  const queryClient = useMemo(() => new QueryClient(), []);

  // hack to silence console warnings in OSS
  if (!(window as any).__databricks_mfe_rpc) {
    Object.defineProperty(window, '__databricks_mfe_rpc', {
      configurable: false,
      writable: false,
      value: {
        // mock all safex calls to return their default value
        makeCall: (id: string, args: any) => {
          if (args?.flagName in FLAG_OVERRIDES) {
            return FLAG_OVERRIDES[args?.flagName];
          }

          return args?.defaultValue;
        },
        hasHandlerFor: () => true,
        registerHandler: () => {},
        unregisterHandler: () => {},
      },
    });
  }

  return (
    <React.Suspense fallback={null}>
      <DesignSystemProviders>
        <IntlProvider locale="en">
          <QueryClientProvider client={queryClient}>
            <LazyModelTraceExplorer />
          </QueryClientProvider>
        </IntlProvider>
      </DesignSystemProviders>
    </React.Suspense>
  );
};

ReactDOM.render(<AppComponent />, document.getElementById('root'));

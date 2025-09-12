import React, { useMemo } from 'react';
import ReactDOM from 'react-dom';

import { IntlProvider } from '@databricks/i18n';
import { SupportsDuBoisThemes } from '@databricks/web-shared/design-system';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';

import './index.css';

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

// eslint-disable-next-line no-restricted-syntax
ReactDOM.render(<AppComponent />, document.getElementById('root'));

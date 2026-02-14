import React, { useMemo } from 'react';
import ReactDOM from 'react-dom';

import { IntlProvider } from '@databricks/i18n';
import { SupportsDuBoisThemes } from '../../design-system/SupportsDuBoisThemes';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';

import './index.css';

const LazyModelTraceExplorer = React.lazy(() =>
  import('../index').then((module) => ({
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

export const AppComponent = () => {
  const queryClient = useMemo(() => new QueryClient(), []);

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

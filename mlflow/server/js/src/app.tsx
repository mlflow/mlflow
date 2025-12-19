import React, { useCallback, useEffect, useMemo } from 'react';
import { ApolloProvider } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { RawIntlProvider } from 'react-intl';

import 'font-awesome/css/font-awesome.css';
import './index.css';

import { ApplyGlobalStyles, DesignSystemEventProvider } from '@databricks/design-system';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';
import { Provider } from 'react-redux';
import store from './store';
import { useI18nInit } from './i18n/I18nUtils';
import { DesignSystemContainer } from './common/components/DesignSystemContainer';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { createApolloClient } from './graphql/client';
import { LegacySkeleton } from '@databricks/design-system';
// eslint-disable-next-line no-useless-rename
import { MlflowRouter as MlflowRouter } from './MlflowRouter';
import { useMLflowDarkTheme } from './common/hooks/useMLflowDarkTheme';
import { DarkThemeProvider } from './common/contexts/DarkThemeContext';
import { telemetryClient } from './telemetry';

export function MLFlowRoot() {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const intl = useI18nInit();
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const apolloClient = useMemo(() => createApolloClient(), []);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const queryClient = useMemo(() => new QueryClient(), []);

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [isDarkTheme, setIsDarkTheme, MlflowThemeGlobalStyles] = useMLflowDarkTheme();

  const logObservabilityEvent = useCallback((event: any) => {
    telemetryClient.logEvent(event);
  }, []);

  if (!intl) {
    return (
      <DesignSystemContainer>
        <LegacySkeleton />
      </DesignSystemContainer>
    );
  }

  return (
    <ApolloProvider client={apolloClient}>
      <RawIntlProvider value={intl} key={intl.locale}>
        <Provider store={store}>
          <DesignSystemEventProvider callback={logObservabilityEvent}>
            <DesignSystemContainer isDarkTheme={isDarkTheme}>
              <ApplyGlobalStyles />
              <MlflowThemeGlobalStyles />
              <DarkThemeProvider setIsDarkTheme={setIsDarkTheme}>
                <QueryClientProvider client={queryClient}>
                  <MlflowRouter />
                </QueryClientProvider>
              </DarkThemeProvider>
            </DesignSystemContainer>
          </DesignSystemEventProvider>
        </Provider>
      </RawIntlProvider>
    </ApolloProvider>
  );
}

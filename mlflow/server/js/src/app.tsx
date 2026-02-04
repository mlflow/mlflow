import React, { useCallback, useEffect, useMemo, useRef } from 'react';
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
import { ServerFeaturesProvider, SERVER_FEATURES_QUERY_KEY } from './common/utils/ServerFeaturesContext';
import { subscribeToWorkspaceChanges } from './workspaces/utils/WorkspaceUtils';

export function MLFlowRoot() {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const intl = useI18nInit();

  // eslint-disable-next-line react-hooks/rules-of-hooks
  // Create clients once - we'll clear caches on workspace changes instead of recreating
  const apolloClient = useMemo(() => createApolloClient(), []);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const queryClient = useMemo(() => new QueryClient(), []);

  // Reset caches when workspace changes instead of recreating clients
  useEffect(() => {
    const unsubscribe = subscribeToWorkspaceChanges(() => {
      // Reset React Query cache except server features (they don't change with workspace)
      // resetQueries removes cached data AND triggers refetch
      queryClient.resetQueries({
        predicate: (query) => {
          // Keep server features cached
          return query.queryKey[0] !== SERVER_FEATURES_QUERY_KEY[0];
        },
      });

      // Clear Apollo cache
      apolloClient.clearStore();
    });

    return unsubscribe;
  }, [queryClient, apolloClient]);

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
                  <ServerFeaturesProvider>
                    <MlflowRouter />
                  </ServerFeaturesProvider>
                </QueryClientProvider>
              </DarkThemeProvider>
            </DesignSystemContainer>
          </DesignSystemEventProvider>
        </Provider>
      </RawIntlProvider>
    </ApolloProvider>
  );
}

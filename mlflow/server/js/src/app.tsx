import React, { useEffect, useMemo, useState } from 'react';
import { ApolloProvider } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { RawIntlProvider } from 'react-intl';

import 'font-awesome/css/font-awesome.css';
import './index.css';

import { ApplyGlobalStyles } from '@databricks/design-system';
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
import '@patternfly/patternfly/patternfly.css';
import {
  DEFAULT_WORKSPACE_NAME,
  getCurrentWorkspace,
  subscribeToWorkspaceChanges,
} from './common/utils/WorkspaceUtils';

export function MLFlowRoot() {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const intl = useI18nInit();
  const [workspaceKey, setWorkspaceKey] = useState(() => getCurrentWorkspace() ?? DEFAULT_WORKSPACE_NAME);

  useEffect(() => {
    return subscribeToWorkspaceChanges((workspace) => {
      setWorkspaceKey(workspace ?? DEFAULT_WORKSPACE_NAME);
    });
  }, []);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  // Recreate clients when workspace changes to clear caches
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const apolloClient = useMemo(() => createApolloClient(), [workspaceKey]);
  // eslint-disable-next-line react-hooks/rules-of-hooks
  // Recreate clients when workspace changes to clear caches
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const queryClient = useMemo(() => new QueryClient(), [workspaceKey]);

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [isDarkTheme, setIsDarkTheme, MlflowThemeGlobalStyles] = useMLflowDarkTheme();

  if (!intl) {
    return (
      <DesignSystemContainer>
        <LegacySkeleton />
      </DesignSystemContainer>
    );
  }

  return (
    <ApolloProvider client={apolloClient} key={workspaceKey}>
      <RawIntlProvider value={intl} key={intl.locale}>
        <Provider store={store}>
          <DesignSystemContainer isDarkTheme={isDarkTheme}>
            <ApplyGlobalStyles />
            <MlflowThemeGlobalStyles />
            <QueryClientProvider key={workspaceKey} client={queryClient}>
              <MlflowRouter key={workspaceKey} isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
            </QueryClientProvider>
          </DesignSystemContainer>
        </Provider>
      </RawIntlProvider>
    </ApolloProvider>
  );
}

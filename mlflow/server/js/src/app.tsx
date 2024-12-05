import React, { useMemo } from 'react';
import { ApolloProvider } from '@apollo/client';
import { RawIntlProvider } from 'react-intl';
import './index.css';
import { ApplyGlobalStyles } from '@databricks/design-system';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';
import { Provider } from 'react-redux';
import store from './store';
import { useI18nInit } from './i18n/I18nUtils';
import { DesignSystemContainer } from './common/components/DesignSystemContainer';
import { ConfigProvider } from 'antd';
import { createApolloClient } from './graphql/client';
import { LegacySkeleton } from '@databricks/design-system';
// eslint-disable-next-line no-useless-rename
import { MlflowRouter as MlflowRouter } from './MlflowRouter';
import { useMLflowDarkTheme } from './common/hooks/useMLflowDarkTheme';

export function MLFlowRoot() {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const intl = useI18nInit();
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const apolloClient = useMemo(() => createApolloClient(), []);

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
    <ApolloProvider client={apolloClient}>
      <RawIntlProvider value={intl} key={intl.locale}>
        <Provider store={store}>
          <DesignSystemContainer isDarkTheme={isDarkTheme}>
            <ApplyGlobalStyles />
            <MlflowThemeGlobalStyles />
            <ConfigProvider prefixCls="ant">
              <MlflowRouter isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
            </ConfigProvider>
          </DesignSystemContainer>
        </Provider>
      </RawIntlProvider>
    </ApolloProvider>
  );
}

import React from 'react';
import { ApolloProvider } from '@apollo/client';
import { IntlProvider } from 'react-intl';
import './index.css';
import { ApplyGlobalStyles } from '@databricks/design-system';
import '@databricks/design-system/dist/index.css';
import '@databricks/design-system/dist/index-dark.css';
import App from './experiment-tracking/components/App';
import { Provider } from 'react-redux';
import store from './store';
import { useI18nInit } from './i18n/I18nUtils';
import { DesignSystemContainer } from './common/components/DesignSystemContainer';
import { ConfigProvider } from 'antd';
import { LegacySkeleton } from '@databricks/design-system';
import { shouldUsePathRouting } from './common/utils/FeatureUtils';
// eslint-disable-next-line no-useless-rename
import { MlflowRouter as MlflowRouter } from './MlflowRouter';
import { useMLflowDarkTheme } from './common/hooks/useMLflowDarkTheme';

export function MLFlowRoot() {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const i18n = useI18nInit();

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [isDarkTheme, setIsDarkTheme, MlflowThemeGlobalStyles] = useMLflowDarkTheme();

  if (!i18n) {
    return (
      <DesignSystemContainer>
        <LegacySkeleton />
      </DesignSystemContainer>
    );
  }

  const { locale, messages } = i18n;

  return (
    <IntlProvider locale={locale} messages={messages}>
      <Provider store={store}>
        <DesignSystemContainer isDarkTheme={isDarkTheme}>
          <ApplyGlobalStyles />
          <MlflowThemeGlobalStyles />
          <ConfigProvider prefixCls="ant">
            {shouldUsePathRouting() ? (
              <MlflowRouter isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
            ) : (
              <App isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
            )}
          </ConfigProvider>
        </DesignSystemContainer>
      </Provider>
    </IntlProvider>
  );
}

import React from 'react';
import { IntlProvider } from 'react-intl';
import './index.css';
import '@databricks/design-system/dist/index.css';
import App from './experiment-tracking/components/App';
import { Provider } from 'react-redux';
import { DesignSystemProvider } from '@databricks/design-system';
import store from './store';
import { injectGlobal } from 'emotion';
import { accessibilityOverrides } from './common/styles/accessibility-overrides';
import { I18nUtils } from './i18n/I18nUtils';

export function init() {
  injectGlobal({ ...accessibilityOverrides });
}

export function MLFlowRoot() {
  const { locale, messages } = I18nUtils.getIntlProviderParams();
  return (
    <IntlProvider locale={locale} messages={messages}>
      <Provider store={store}>
        <DesignSystemProvider>
          <App />
        </DesignSystemProvider>
      </Provider>
    </IntlProvider>
  );
}

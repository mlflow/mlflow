import React from 'react';
import ReactDOM from 'react-dom';
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

I18nUtils.initI18n().then(() => {
  const { locale, messages } = I18nUtils.getIntlProviderParams();
  const root = (
    <IntlProvider locale={locale} messages={messages}>
      <Provider store={store}>
        <DesignSystemProvider>
          <App />
        </DesignSystemProvider>
      </Provider>
    </IntlProvider>
  );
  ReactDOM.render(root, document.getElementById('root'));
  injectGlobal({ ...accessibilityOverrides });
});

const windowOnError = (message, source, lineno, colno, error) => {
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

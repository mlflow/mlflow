import React, { useEffect } from 'react';
import { IntlProvider } from 'react-intl';
import './index.css';
import '@databricks/design-system/dist/index.css';
import App from './experiment-tracking/components/App';
import { Provider } from 'react-redux';
import store from './store';
import { I18nUtils } from './i18n/I18nUtils';
import { DesignSystemContainer } from './common/components/DesignSystemContainer';
import { ConfigProvider } from 'antd';
import { BrowserRouter as Router} from 'react-router-dom';

export function MLFlowRoot() {
  const { locale, messages } = I18nUtils.getIntlProviderParams();

  return (
    <Router>
    <IntlProvider locale={locale} messages={messages}>
      <Provider store={store}>
        <DesignSystemContainer flags={{ USE_NEW_ICONS: true }}>
          <ConfigProvider prefixCls='ant'>
            <App />
          </ConfigProvider>
        </DesignSystemContainer>
      </Provider>
    </IntlProvider>
    </Router>);
}

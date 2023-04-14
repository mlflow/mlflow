import React from 'react';
import ReactDOM from 'react-dom';
import { I18nUtils } from './i18n/I18nUtils';
import { MLFlowRoot } from './app';

I18nUtils.initI18n().then(() => {
  ReactDOM.render(<MLFlowRoot />, document.getElementById('root'));
});

const windowOnError = (message: any, source: any, lineno: any, colno: any, error: any) => {
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

import React from 'react';
import ReactDOM from 'react-dom';
import { I18nUtils } from './i18n/I18nUtils';
import { init, MLFlowRoot } from './app';

I18nUtils.initI18n().then(() => {
  init();
  ReactDOM.render(<MLFlowRoot />, document.getElementById('root'));
});

const windowOnError = (message, source, lineno, colno, error) => {
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

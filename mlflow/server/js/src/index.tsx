import React from 'react';
import ReactDOM from 'react-dom';
import { MLFlowRoot } from './app';
import { AlertUtils } from '@databricks/web-shared/alert-utils';

ReactDOM.render(<MLFlowRoot />, document.getElementById('root'));

const windowOnError = (message: Event | string, source?: string, lineno?: number, colno?: number, error?: Error) => {
  AlertUtils.log('Uncaught window error', { error, message });
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

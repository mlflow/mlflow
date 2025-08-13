import React from 'react';
import ReactDOM from 'react-dom';
import { MLFlowRoot } from './app';

ReactDOM.render(<MLFlowRoot />, document.getElementById('root'));

const windowOnError = (message: Event | string, source?: string, lineno?: number, colno?: number, error?: Error) => {
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

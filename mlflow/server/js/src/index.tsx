/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import ReactDOM from 'react-dom';
import { I18nUtils } from './i18n/I18nUtils';
import { MLFlowRoot } from './app';

ReactDOM.render(<MLFlowRoot />, document.getElementById('root'));

const windowOnError = (message: any, source: any, lineno: any, colno: any, error: any) => {
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;

import 'react-app-polyfill/ie9'; // For IE 9-11 support

import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './components/App';
import registerServiceWorker from './registerServiceWorker';
import { setupAjaxHeaders } from './setupAjaxHeaders';
import { Provider } from 'react-redux';
import store from './Store';

setupAjaxHeaders();

const root = (
  <Provider store={store}>
    <App/>
  </Provider>
);
ReactDOM.render(root, document.getElementById('root'));

registerServiceWorker();

import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';

import { rootReducer } from './experiment-tracking/reducers/Reducers';

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

// exported for testing
export const makeStore = (preLoadedState = {}) => {
  return createStore(
    rootReducer,
    preLoadedState,
    composeEnhancers(applyMiddleware(thunk, promiseMiddleware())),
  );
};
const store = makeStore();

export default store;

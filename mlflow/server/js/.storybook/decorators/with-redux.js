import { Provider } from 'react-redux';
import { applyMiddleware } from 'redux';
import { createStore } from 'redux';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

const identityReducer = (store) => store;

/**
 * Adds redux capabilities to stories by wrapping the story
 * with redux provider
 *
 * Basic usage that provides identity reducer and empty state:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withRedux: true
 *   }
 * };
 *
 * Usage with setting custom reducer and/or initial state:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withRedux: {
 *       reducer: (store) => store,
 *       initialState: { foo: 'bar' }
 *     },
 *   },
 * };
 */
export const withReduxDecorator = (Story, { parameters }) => {
  if (parameters.withRedux) {
    const createStoreProps = typeof parameters.withRedux === 'object' ? parameters.withRedux : {};
    return (
      <Provider
        store={createStore(
          createStoreProps.reducer || identityReducer,
          { ...(createStoreProps.initialState || {}) },
          applyMiddleware(thunk, promiseMiddleware()),
        )}
      >
        <Story />
      </Provider>
    );
  }

  return <Story />;
};

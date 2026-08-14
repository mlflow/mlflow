import type { DeepPartial } from 'redux';

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

import React, { useRef } from 'react';
import { DEFAULT_LOCALE } from '../../i18n/loadMessages';
import type { ReduxState } from '../../redux-types';
import { Provider } from 'react-redux';

export const NOOP = (...args: any[]) => {};

export function deepFreeze(o: any) {
  Object.freeze(o);
  Object.getOwnPropertyNames(o).forEach((prop) => {
    if (
      o.hasOwnProperty(prop) &&
      o[prop] !== null &&
      (typeof o[prop] === 'object' || typeof o[prop] === 'function') &&
      !Object.isFrozen(o[prop])
    ) {
      deepFreeze(o[prop]);
    }
  });
  return o;
}

export const defaultProviderProps = {
  locale: DEFAULT_LOCALE,
  defaultLocale: DEFAULT_LOCALE,
  messages: {},
};

/**
 * A simple seedable PRNG, used e.g. to replace Math.random() for deterministic testing.
 * Taken from https://gist.github.com/blixt/f17b47c62508be59987b
 */
export const createPrng = (seed = 1000) => {
  let _seed = seed % 2147483647;
  if (_seed <= 0) _seed += 2147483646;

  const next = () => {
    return (_seed = (_seed * 16807) % 2147483647);
  };

  return () => (next() - 1) / 2147483646;
};

export const MockedReduxStoreProvider = ({
  state = {},
  children,
}: {
  state?: DeepPartial<ReduxState>;
  children: React.ReactNode;
}) => {
  const store = useRef(configureStore([thunk, promiseMiddleware()])(state));
  return <Provider store={store.current}>{children}</Provider>;
};

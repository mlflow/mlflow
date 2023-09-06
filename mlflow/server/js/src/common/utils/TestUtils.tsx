import { render as rtlRender } from '@testing-library/react';

import React from 'react';
import { IntlProvider } from 'react-intl';
import { shallow, mount } from 'enzyme';
import { DEFAULT_LOCALE } from '../../i18n/loadMessages';

export const NOOP = () => {};

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

const defaultProvderProps = {
  locale: DEFAULT_LOCALE,
  defaultLocale: DEFAULT_LOCALE,
  messages: {},
};

export function mountWithIntl(node: React.ReactElement, providerProps = {}) {
  return mount(node, {
    wrappingComponent: IntlProvider,
    wrappingComponentProps: {
      ...defaultProvderProps,
      ...providerProps,
    },
  });
}

export function shallowWithIntl(node: React.ReactElement, providerProps = {}) {
  const mergedProviderProps = {
    ...defaultProvderProps,
    ...providerProps,
  };
  return shallow(<IntlProvider {...mergedProviderProps}>{node}</IntlProvider>).dive();
}

export function shallowWithInjectIntl(node: React.ReactElement, providerProps = {}) {
  return shallowWithIntl(node, providerProps).dive().dive().dive();
}

function renderWithIntl(ui: React.ReactElement, renderOptions = {}, providerProps = {}) {
  const mergedProviderProps = {
    ...defaultProvderProps,
    ...providerProps,
  };
  const wrapper: React.JSXElementConstructor<{ children: React.ReactElement }> = ({ children }) => (
    <IntlProvider {...mergedProviderProps}>{children}</IntlProvider>
  );

  return rtlRender(ui, { wrapper, ...renderOptions });
}

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

export * from '@testing-library/react';
export { renderWithIntl };

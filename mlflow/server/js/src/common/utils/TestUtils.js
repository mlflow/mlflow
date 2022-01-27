import React from 'react';
import { IntlProvider } from 'react-intl';
import $ from 'jquery';
import { shallow, mount } from 'enzyme';
import { DEFAULT_LOCALE } from '../../i18n/loadMessages';

export const NOOP = () => {};

export function deepFreeze(o) {
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

// Replaces AJAX calls with a mock function that returns a nonsense result. This is sufficient
// for testing connected components that would otherwise attempt to make real REST calls, but
// which have no reducers associated -- so the result of the call is thrown out. We mostly
// just want to prevent actual connection attempts from being made and throwing spurious errors.
export function mockAjax() {
  $.ajax = jest.fn().mockImplementation(() => {
    return Promise.resolve({ value: '' });
  });
}

const defaultProvderProps = {
  locale: DEFAULT_LOCALE,
  defaultLocale: DEFAULT_LOCALE,
  messages: {},
};

export function mountWithIntl(node, providerProps = {}) {
  return mount(node, {
    wrappingComponent: IntlProvider,
    wrappingComponentProps: {
      ...defaultProvderProps,
      ...providerProps,
    },
  });
}

export function shallowWithIntl(node, providerProps = {}) {
  const mergedProviderProps = {
    ...defaultProvderProps,
    ...providerProps,
  };
  return shallow(<IntlProvider {...mergedProviderProps}>{node}</IntlProvider>).dive();
}

export function shallowWithInjectIntl(node, providerProps = {}) {
  return shallowWithIntl(node, providerProps)
    .dive()
    .dive()
    .dive();
}

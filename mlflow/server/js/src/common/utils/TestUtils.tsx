/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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

export function mountWithIntl(node: any, providerProps = {}) {
  return mount(node, {
    wrappingComponent: IntlProvider,
    wrappingComponentProps: {
      ...defaultProvderProps,
      ...providerProps,
    },
  });
}

export function shallowWithIntl(node: any, providerProps = {}) {
  const mergedProviderProps = {
    ...defaultProvderProps,
    ...providerProps,
  };
  return shallow(<IntlProvider {...mergedProviderProps}>{node}</IntlProvider>).dive();
}

export function shallowWithInjectIntl(node: any, providerProps = {}) {
  return shallowWithIntl(node, providerProps).dive().dive().dive();
}

import React from 'react';
import { IntlProvider } from 'react-intl';
import { mount, shallow } from 'enzyme';
import { defaultProviderProps } from './TestUtils';

export function mountWithIntl(node: React.ReactElement, providerProps = {}) {
  return mount(node, {
    wrappingComponent: IntlProvider,
    wrappingComponentProps: {
      ...defaultProviderProps,
      ...providerProps,
    },
  });
}
export function shallowWithIntl(node: React.ReactElement, providerProps = {}) {
  const mergedProviderProps = {
    ...defaultProviderProps,
    ...providerProps,
  };
  return shallow(<IntlProvider {...mergedProviderProps}>{node}</IntlProvider>).dive();
}
export function shallowWithInjectIntl(node: React.ReactElement, providerProps = {}) {
  return shallowWithIntl(node, providerProps).dive().dive().dive();
}

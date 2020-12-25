import React from 'react';
import { shallow } from 'enzyme';
import AppErrorBoundary from './AppErrorBoundary';
import { SupportPageUrl } from '../../constants';

describe('AppErrorBoundary', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = { children: 'testChild' };
    wrapper = shallow(<AppErrorBoundary {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.text()).toEqual('testChild');
    expect(wrapper.find('.error-image').length).toBe(0);
  });

  test('test componentDidCatch causes error message to render', () => {
    const instance = wrapper.instance();
    instance.componentDidCatch('testError', 'testInfo');
    instance.forceUpdate();
    expect(wrapper.find('.error-image').length).toBe(1);
    expect(wrapper.text()).not.toMatch('testChild');
    expect(wrapper.find({ href: SupportPageUrl }).length).toBe(1);
  });
});

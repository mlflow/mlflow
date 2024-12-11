/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { SectionErrorBoundary } from './SectionErrorBoundary';
import { SupportPageUrl } from '../../constants';

describe('SectionErrorBoundary', () => {
  let wrapper: any;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = { children: 'testChild' };
    wrapper = shallow(<SectionErrorBoundary {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.text()).toEqual('testChild');
    expect(wrapper.find('i.icon-fail').length).toBe(0);
  });

  test('test componentDidCatch causes error message to render', () => {
    const instance = wrapper.instance();
    instance.componentDidCatch('testError', 'testInfo');
    instance.forceUpdate();
    expect(wrapper.find('i.icon-fail').length).toBe(1);
    expect(wrapper.text()).not.toMatch('testChild');
    expect(wrapper.find({ href: SupportPageUrl }).length).toBe(1);
  });

  test('should show error if showServerError prop passed in', () => {
    const withShowServerError = shallow(<SectionErrorBoundary {...minimalProps} showServerError />);
    const instance = withShowServerError.instance();
    instance.componentDidCatch(new Error('some error message'));
    instance.forceUpdate();
    expect(withShowServerError.text()).toContain('some error message');
  });
});

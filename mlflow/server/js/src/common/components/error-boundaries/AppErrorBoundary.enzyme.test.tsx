/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mount, shallow } from 'enzyme';
import AppErrorBoundary from './AppErrorBoundary';
import { SupportPageUrl } from '../../constants';
import Utils from '../../utils/Utils';

describe('AppErrorBoundary', () => {
  let wrapper: any;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      children: <div data-testid="child-component">testChild</div>,
    };
    wrapper = shallow(<AppErrorBoundary {...minimalProps} />).dive();
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.find('[data-testid="child-component"]')).toHaveLength(1);
    expect(wrapper.find('.mlflow-error-image').length).toBe(0);
  });

  test('componentDidCatch causes error message to render', () => {
    const instance = wrapper.instance();
    instance.componentDidCatch('testError', 'testInfo');
    instance.forceUpdate();
    expect(wrapper.find('.mlflow-error-image').length).toBe(1);
    expect(wrapper.text()).not.toMatch('testChild');
    expect(wrapper.find({ href: SupportPageUrl }).length).toBe(1);
  });
  test('register its notifications API in global utils', () => {
    jest.spyOn(Utils, 'registerNotificationsApi').mockImplementation(() => {});
    mount(<AppErrorBoundary {...minimalProps} />);
    expect(Utils.registerNotificationsApi).toHaveBeenCalledTimes(1);
  });
});

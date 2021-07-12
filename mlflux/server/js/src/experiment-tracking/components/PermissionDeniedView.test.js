import React from 'react';
import { shallow } from 'enzyme';
import { PermissionDeniedView } from './PermissionDeniedView';

describe('PermissionDeniedView', () => {
  let wrapper;
  let minimalProps;
  const mockErrorMessage = 'This is a mock error message';
  const defaultMessage = 'The current user does not have permission to view this page.';

  beforeEach(() => {
    minimalProps = { errorMessage: mockErrorMessage };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<PermissionDeniedView {...minimalProps} />);
  });

  test('should render correct h2 text when error message is passed', () => {
    wrapper = shallow(<PermissionDeniedView {...minimalProps} />);
    expect(wrapper.childAt(2).text()).toBe(mockErrorMessage);
  });

  test('should render default message in h2 when no error message is passed', () => {
    wrapper = shallow(<PermissionDeniedView {...minimalProps} />);
    wrapper.setProps({ errorMessage: '' });
    expect(wrapper.childAt(2).text()).toBe(defaultMessage);
  });
});

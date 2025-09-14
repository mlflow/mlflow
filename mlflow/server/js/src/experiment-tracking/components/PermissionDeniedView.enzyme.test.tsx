/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { PermissionDeniedView } from './PermissionDeniedView';

describe('PermissionDeniedView', () => {
  let wrapper;
  let minimalProps: any;
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
    wrapper = shallow(<PermissionDeniedView />);
    expect(wrapper.find('[data-testid="mlflow-error-message"]').text()).toBe(defaultMessage);
  });
});

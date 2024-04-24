/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { ErrorModalWithIntl } from './ErrorModal';
import { Modal } from '@databricks/design-system';

describe('ErrorModalImpl', () => {
  let wrapper: any;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      text: 'Error popup content',
    };
    wrapper = mountWithIntl(<ErrorModalWithIntl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(Modal).length).toBe(1);
  });
});

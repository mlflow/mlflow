import React from 'react';
import { mountWithIntl } from '../../../common/utils/TestUtils';
import { ErrorModalWithIntl } from './ErrorModal';
import { Modal } from 'antd';

describe('ErrorModalImpl', () => {
  let wrapper;
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

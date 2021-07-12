import React from 'react';
import { shallow } from 'enzyme';
import { ErrorModalImpl } from './ErrorModal';
import ReactModal from 'react-modal';

describe('ErrorModalImpl', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      text: 'Error popup content',
    };
    wrapper = shallow(<ErrorModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ErrorModalImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find(ReactModal).length).toBe(1);
  });
});

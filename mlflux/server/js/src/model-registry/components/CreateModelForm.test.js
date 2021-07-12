import React from 'react';
import { shallow } from 'enzyme';
import { CreateModelForm } from './CreateModelForm';

describe('Render test', () => {
  const minimalProps = {
    visible: true,
    form: { getFieldDecorator: jest.fn(() => (c) => c) },
  };

  test('should render with minimal props without exploding', () => {
    const wrapper = shallow(<CreateModelForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

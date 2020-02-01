import React from 'react';
import { shallow } from 'enzyme';
import { InputFormView } from './InputFormView';

describe('Render test', () => {
  const defaultProps = {
    type: 'run',
    name: 'Test',
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn(opts => c => c) },
  };

  it('renders with props', () => {
    const wrapper = shallow(<InputFormView {...defaultProps} />);
    expect(wrapper.length).toBe(1);
  });
});

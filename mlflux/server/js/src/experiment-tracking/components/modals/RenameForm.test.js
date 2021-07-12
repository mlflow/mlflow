import React from 'react';
import { shallow } from 'enzyme';
import { RenameForm } from './RenameForm';

describe('Render test', () => {
  const minimalProps = {
    type: 'run',
    name: 'Test',
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c) => c) },
  };

  it('should render with minimal props without exploding', () => {
    const wrapper = shallow(<RenameForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

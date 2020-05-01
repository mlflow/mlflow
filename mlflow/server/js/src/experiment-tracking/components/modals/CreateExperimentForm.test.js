import React from 'react';
import { shallow } from 'enzyme';
import { CreateExperimentForm } from './CreateExperimentForm';

describe('Render test', () => {
  const minimalProps = {
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c) => c) },
  };

  it('should render with minimal props without exploding', () => {
    const wrapper = shallow(<CreateExperimentForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

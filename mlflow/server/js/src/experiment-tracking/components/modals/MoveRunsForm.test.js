import React from 'react';
import { shallow } from 'enzyme';
import { MoveRunsForm } from './MoveRunsForm';

describe('Render test', () => {
  const minimalProps = {
    visible: true,
    experimentList: [{ experiment_id: '1' }, { experiment_id: '2' }],
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c) => c) },
  };

  it('should render with minimal props without exploding', () => {
    const wrapper = shallow(<MoveRunsForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

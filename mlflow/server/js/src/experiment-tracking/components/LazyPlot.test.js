import { shallow } from 'enzyme';
import React from 'react';
import { LazyPlot } from './LazyPlot';

describe('LazyPlot', () => {
  it('should render with minimal props without exploding', () => {
    const wrapper = shallow(<LazyPlot />);
    expect(wrapper.length).toBe(1);
  });
});

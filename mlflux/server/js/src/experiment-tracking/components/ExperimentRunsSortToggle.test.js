import React from 'react';
import { shallow } from 'enzyme';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';

describe('ExperimentRunsSortToggle', () => {
  let wrapper;

  beforeEach(() => {
    wrapper = shallow(<ExperimentRunsSortToggle />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
  });
});

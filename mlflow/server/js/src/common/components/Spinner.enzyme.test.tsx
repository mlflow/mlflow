import React from 'react';
import { shallow } from 'enzyme';
import { Spinner } from './Spinner';

describe('Spinner', () => {
  let wrapper;

  test('should render with no props without exploding', () => {
    wrapper = shallow(<Spinner />);
    expect(wrapper.length).toBe(1);
  });
});

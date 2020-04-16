import React from 'react';
import { shallow } from 'enzyme';
import { Spinner } from './Spinner';

describe('Spinner', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = { showImmediately: false };
  });

  test('should render with no props without exploding', () => {
    wrapper = shallow(<Spinner />);
    expect(wrapper.length).toBe(1);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<Spinner {...minimalProps} />);
    expect(wrapper.find('.Spinner')).toHaveLength(1);
  });

  test('should render when showImmediately is true', () => {
    wrapper = shallow(<Spinner {...minimalProps} />);
    wrapper.setProps({ showImmediately: true });
    expect(wrapper.find('.Spinner-immediate')).toHaveLength(1);
  });
});

import React from 'react';
import { shallow } from 'enzyme';
import { CollapsibleTagsCell } from './CollapsibleTagsCell';

describe('unit tests', () => {
  let wrapper;
  const minimalProps = {

  };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<CollapsibleTagsCell {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

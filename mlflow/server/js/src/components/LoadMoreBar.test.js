import React from 'react';
import { shallow } from 'enzyme';
import { LoadMoreBar } from './LoadMoreBar';

describe('unit tests', () => {
  let wrapper;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      height: 37,
      width: 1000,
      borderStyle: '',
      loadingMore: false,
      onLoadMore: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<LoadMoreBar {...mininumProps}/>);
    expect(wrapper.length).toBe(1);
  });
});

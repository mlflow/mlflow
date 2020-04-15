import React from 'react';
import { shallow } from 'enzyme';
import { LoadMoreBar } from './LoadMoreBar';

describe('unit tests', () => {
  let wrapper;
  let mininumProps;

  beforeEach(() => {
    mininumProps = {
      loadingMore: false,
      onLoadMore: jest.fn(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<LoadMoreBar {...mininumProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('should render load-more button by default', () => {
    const props = { ...mininumProps, loadingMore: false };
    wrapper = shallow(<LoadMoreBar {...props}/>);
    expect(wrapper.find('.load-more-button').length).toBe(1);
  });

  test('should render loading icon when loadingMore is true', () => {
    const props = { ...mininumProps, loadingMore: true };
    wrapper = shallow(<LoadMoreBar {...props}/>);
    expect(wrapper.find('.loading-more-wrapper').length).toBe(1);
  });
});

import React from 'react';
import { LoadMoreBar } from './LoadMoreBar';
import { shallowWithInjectIntl } from '../../common/utils/TestUtils';

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
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...mininumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render load-more button by default', () => {
    const props = { ...mininumProps, loadingMore: false };
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...props} />);
    expect(wrapper.find('.load-more-button').length).toBe(1);
  });

  test('should render loading icon when loadingMore is true', () => {
    const props = { ...mininumProps, loadingMore: true };
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...props} />);
    expect(wrapper.find('.loading-more-wrapper').length).toBe(1);
  });

  test('should enable button and not show tooltip when disableButton is false', () => {
    const props = { ...mininumProps, disableButton: false };
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...props} />);
    expect(wrapper.find('.load-more-button-tooltip').length).toBe(0);
    expect(wrapper.find('.load-more-button').props().disabled).toBe(false);
  });

  test('should disable button and show tooltip when disableButton is true', () => {
    const props = { ...mininumProps, disableButton: true };
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...props} />);
    expect(wrapper.find('.load-more-button-disabled-tooltip').length).toBe(1);
    expect(wrapper.find('.load-more-button').props().disabled).toBe(true);
  });

  test('should show nested info tooltip when nestChildren is true', () => {
    const props = { ...mininumProps, disableButton: false, nestChildren: true };
    wrapper = shallowWithInjectIntl(<LoadMoreBar {...props} />);
    expect(wrapper.find('.load-more-button-nested-info-tooltip').length).toBe(1);
    expect(wrapper.find('.load-more-button').props().disabled).toBe(false);
  });
});

import React from 'react';
import { mount } from 'enzyme';
import { SimplePagination } from './SimplePagination';

describe('SimplePagination', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      currentPage: 3,
      isLastPage: false,
      onClickNext: jest.fn(),
      onClickPrev: jest.fn(),
      getSelectedPerPageSelection: () => 25,
    };
  });

  test('should render button in disabled state based on page', () => {
    wrapper = mount(<SimplePagination {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    // prev and next buttons are rendered and not disabled
    expect(wrapper.find('.ant-pagination-prev').prop('aria-disabled')).toBe(false);
    expect(wrapper.find('.ant-pagination-next').prop('aria-disabled')).toBe(false);

    // first page => prev button is disabled
    wrapper.setProps({ currentPage: 1, isLastPage: false });
    expect(wrapper.find('.ant-pagination-prev').prop('aria-disabled')).toBe(true);
    expect(wrapper.find('.ant-pagination-next').prop('aria-disabled')).toBe(false);

    // first and last page => both buttons are disabled
    wrapper.setProps({ currentPage: 1, isLastPage: true });
    // check that 1 disabled shows up for next page btn
    expect(wrapper.find('.ant-pagination-prev').prop('aria-disabled')).toBe(true);
    expect(wrapper.find('.ant-pagination-next').prop('aria-disabled')).toBe(true);

    // last page => next button is disabled
    wrapper.setProps({ currentPage: 2, isLastPage: true });
    expect(wrapper.find('.ant-pagination-prev').prop('aria-disabled')).toBe(false);
    expect(wrapper.find('.ant-pagination-next').prop('aria-disabled')).toBe(true);
  });
});

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mount } from 'enzyme';
import { SimplePagination } from './SimplePagination';

describe('SimplePagination', () => {
  let wrapper;
  let minimalProps: any;

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
    // TODO: wrap this with DesignSystemProvider (consider rerendering instead of setProps)
    wrapper = mount(<SimplePagination {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    // prev and next buttons are rendered and not disabled
    expect(wrapper.find('[title="Previous Page"]').prop('aria-disabled')).toBe(false);
    expect(wrapper.find('[title="Next Page"]').prop('aria-disabled')).toBe(false);

    // first page => prev button is disabled
    wrapper.setProps({ currentPage: 1, isLastPage: false });
    expect(wrapper.find('[title="Previous Page"]').prop('aria-disabled')).toBe(true);
    expect(wrapper.find('[title="Next Page"]').prop('aria-disabled')).toBe(false);

    // first and last page => both buttons are disabled
    wrapper.setProps({ currentPage: 1, isLastPage: true });
    // check that 1 disabled shows up for next page btn
    expect(wrapper.find('[title="Previous Page"]').prop('aria-disabled')).toBe(true);
    expect(wrapper.find('[title="Next Page"]').prop('aria-disabled')).toBe(true);

    // last page => next button is disabled
    wrapper.setProps({ currentPage: 2, isLastPage: true });
    expect(wrapper.find('[title="Previous Page"]').prop('aria-disabled')).toBe(false);
    expect(wrapper.find('[title="Next Page"]').prop('aria-disabled')).toBe(true);
  });
});

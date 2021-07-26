import { shallow } from 'enzyme';
import { SimplePagination } from './SimplePagination';
import React from 'react';
import { Button } from 'antd';

describe('SimplePagination', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      currentPage: 3,
      isLastPage: false,
      onClickNext: jest.fn(),
      onClickPrev: jest.fn(),
    };
  });

  test('should render button in disabled state based on page', () => {
    wrapper = shallow(<SimplePagination {...minimalProps} />);
    expect(wrapper.length).toBe(1);
    // prev and next buttons are rendered and not disabled
    let buttons = wrapper.find(Button);
    expect(buttons.length).toBe(2);
    expect(buttons.at(0).props().disabled).toBe(false);
    expect(buttons.at(1).props().disabled).toBe(false);

    // first page => prev button is disabled
    wrapper.setProps({ currentPage: 1, isLastPage: false });
    buttons = wrapper.find(Button);
    expect(buttons.length).toBe(2);
    expect(buttons.at(0).props().disabled).toBe(true);
    expect(buttons.at(1).props().disabled).toBe(false);

    // first and last page => both buttons are disabled
    wrapper.setProps({ currentPage: 1, isLastPage: true });
    buttons = wrapper.find(Button);
    // check that 1 disabled shows up for next page btn
    expect(buttons.length).toBe(2);
    expect(buttons.at(0).props().disabled).toBe(true);
    expect(buttons.at(1).props().disabled).toBe(true);

    // last page => next button is disabled
    wrapper.setProps({ currentPage: 2, isLastPage: true });
    buttons = wrapper.find(Button);
    expect(buttons.length).toBe(2);
    expect(buttons.at(0).props().disabled).toBe(false);
    expect(buttons.at(1).props().disabled).toBe(true);
  });
});

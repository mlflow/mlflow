import React from 'react';
import { shallow, mount } from 'enzyme';
import BaggedCell from './BaggedCell';
import { Dropdown } from 'antd';

describe('BaggedCell', () => {
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      keyName: 'key1',
      value: 'value1',
      onSortBy: () => {},
      isParam: false,
      isMetric: true,
      onRemoveBagged: () => {},
    };
  });

  test('should render with minimal props without exploding', () => {
    const wrapper = shallow(<BaggedCell {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should use correct key in sortAscending', () => {
    const mockOnSortByFn = jest.fn();
    const newProps = {
      ...minimalProps,
      onSortBy: mockOnSortByFn,
    };
    const wrapper = shallow(<BaggedCell {...newProps} />);
    const overlay = mount(
      wrapper
        .find(Dropdown)
        .first()
        .props().overlay,
    );
    const sortAscButton = overlay.find('[data-test-id="sort-ascending"]');
    sortAscButton.first().simulate('click');
    expect(mockOnSortByFn.mock.calls[0][0]).toEqual(expect.stringContaining('key1'));
    expect(mockOnSortByFn.mock.calls[0][1]).toBeTruthy(); // ascending
  });

  test('should use correct key in sortDescending', () => {
    const mockOnSortByFn = jest.fn();
    const newProps = {
      ...minimalProps,
      onSortBy: mockOnSortByFn,
    };
    const wrapper = shallow(<BaggedCell {...newProps} />);
    const overlay = mount(
      wrapper
        .find(Dropdown)
        .first()
        .props().overlay,
    );
    const sortDescButton = overlay.find('[data-test-id="sort-descending"]');
    sortDescButton.first().simulate('click');
    expect(mockOnSortByFn.mock.calls[0][0]).toEqual(expect.stringContaining('key1'));
    expect(mockOnSortByFn.mock.calls[0][1]).toBeFalsy(); // descending
  });

  test('should remove bagged', () => {
    const mockOnRemoveBaggedFn = jest.fn();
    const newProps = {
      ...minimalProps,
      onRemoveBagged: mockOnRemoveBaggedFn,
    };
    const wrapper = shallow(<BaggedCell {...newProps} />);
    const overlay = mount(
      wrapper
        .find(Dropdown)
        .first()
        .props().overlay,
    );
    const removeBaggedButton = overlay.find('[data-test-id="remove-bagged"]');
    removeBaggedButton.first().simulate('click');
    expect(mockOnRemoveBaggedFn.mock.calls[0][0]).toBeFalsy();
    expect(mockOnRemoveBaggedFn.mock.calls[0][1]).toEqual(expect.stringContaining('key1'));
  });
});

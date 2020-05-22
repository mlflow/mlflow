import React from 'react';
import { shallow, mount } from 'enzyme';
import { RunsTableCustomHeader, SortByIcon } from './RunsTableCustomHeader';

describe('RunsTableCustomHeader', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {};
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<RunsTableCustomHeader {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render sorting icon correctly', () => {
    const props = {
      ...minimalProps,
      enableSorting: true,
      canonicalSortKey: 'user',
      orderByKey: 'user',
      orderByAsc: true,
      onSortBy: jest.fn(),
    };
    wrapper = mount(<RunsTableCustomHeader {...props} />);
    expect(wrapper.find(SortByIcon).length).toBe(1);
    expect(wrapper.find(SortByIcon).prop('orderByAsc')).toBe(true);

    // should not show sorting icon when sorting is disabled
    props.enableSorting = false;
    wrapper = mount(<RunsTableCustomHeader {...props} />);
    expect(wrapper.find(SortByIcon).length).toBe(0);
  });

  test('should contain child accessibility role since ag-grid has aria parent', () => {
    wrapper = shallow(<RunsTableCustomHeader {...minimalProps} />);
    expect(wrapper.find("[role='columnheader']").length).toBe(1);
  });
});

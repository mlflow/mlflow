/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow, mount } from 'enzyme';
import { RunsTableCustomHeader, SortByIcon } from './RunsTableCustomHeader';

describe('RunsTableCustomHeader', () => {
  let wrapper;
  let minimalProps: any;

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

  test('should handleSortBy correctly', () => {
    const onSortBy = jest.fn();
    const props = {
      ...minimalProps,
      enableSorting: true,
      canonicalSortKey: 'user',
      orderByKey: 'username',
      orderByAsc: false,
      onSortBy,
    };
    wrapper = mount(<RunsTableCustomHeader {...props} />);
    let instance = wrapper.instance();
    instance.handleSortBy();

    expect(onSortBy).toHaveBeenCalledTimes(1);
    expect(onSortBy).toBeCalledWith(props.canonicalSortKey, false);

    props.orderByKey = 'user';
    wrapper = mount(<RunsTableCustomHeader {...props} />);
    instance = wrapper.instance();
    instance.handleSortBy();

    expect(onSortBy).toHaveBeenCalledTimes(2);
    expect(onSortBy).toBeCalledWith(props.canonicalSortKey, true);
  });

  test('should compute the styles based on sort key correctly', () => {
    const style = { backgroundColor: '#e6f7ff' };
    const key = 'user';
    const computedStylesOnSortKey = jest.fn().mockReturnValue(style);
    const props = {
      computedStylesOnSortKey,
      enableSorting: true,
      canonicalSortKey: key,
      orderByKey: key,
      orderByAsc: true,
    };
    wrapper = mount(<RunsTableCustomHeader {...props} />);

    expect(computedStylesOnSortKey).toHaveBeenCalledTimes(1);
    expect(computedStylesOnSortKey).toBeCalledWith(key);

    const containerStyle = wrapper.find('[role="columnheader"]').props().style;
    expect(containerStyle).toHaveProperty('backgroundColor', '#e6f7ff');
  });
});

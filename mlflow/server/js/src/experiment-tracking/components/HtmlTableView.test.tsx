/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { shallow, mount } from 'enzyme';
import React from 'react';
import { LegacyTable } from '@databricks/design-system';
import { HtmlTableView } from './HtmlTableView';

describe('HtmlTableView', () => {
  let wrapper;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      columns: [],
      values: [],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<HtmlTableView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render column and values', () => {
    const props = {
      columns: [
        { title: 'one', dataIndex: 'one' },
        { title: 'two', dataIndex: 'two' },
        { title: 'three', dataIndex: 'three' },
      ],
      values: [
        { key: 'row-one', one: 1, two: 2, three: 3 },
        { key: 'row-two', one: 4, two: 5, three: 6 },
      ],
    };

    wrapper = mount(<HtmlTableView {...props} />);
    const table = wrapper.find(LegacyTable);
    expect(wrapper.find(LegacyTable).length).toBe(1);

    const rows = table.find('tr');
    expect(rows.length).toBe(3);

    const headers = rows.first().find('th');
    expect(headers.at(0).text()).toBe('one');
    expect(headers.at(1).text()).toBe('two');
    expect(headers.at(2).text()).toBe('three');

    const rowOne = rows.at(1).find('td');
    expect(rowOne.at(0).text()).toBe('1');
    expect(rowOne.at(1).text()).toBe('2');
    expect(rowOne.at(2).text()).toBe('3');

    const rowTwo = rows.at(2).find('td');
    expect(rowTwo.at(0).text()).toBe('4');
    expect(rowTwo.at(1).text()).toBe('5');
    expect(rowTwo.at(2).text()).toBe('6');
  });

  test('should render styles', () => {
    const props = {
      ...minimalProps,
      styles: {
        width: 'auto',
        minWidth: '400px',
      },
    };

    wrapper = shallow(<HtmlTableView {...props} />);
    const tableStlye = wrapper.find(LegacyTable).get(0).props.style;
    expect(tableStlye).toHaveProperty('width', 'auto');
    expect(tableStlye).toHaveProperty('minWidth', '400px');
  });
});

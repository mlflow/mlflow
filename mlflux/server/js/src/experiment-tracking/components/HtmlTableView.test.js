import { shallow } from 'enzyme';
import { HtmlTableView } from './HtmlTableView';
import React from 'react';
import { Table } from 'react-bootstrap';

describe('HtmlTableView', () => {
  let wrapper;
  let minimalProps;

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
      columns: ['one', 'two', 'three'],
      values: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    };

    wrapper = shallow(<HtmlTableView {...props} />);
    expect(wrapper.find(Table).length).toBe(1);

    const rows = wrapper.find('tr');
    expect(rows.length).toBe(3);

    const headers = rows.first().children('th');
    expect(headers.at(0).text()).toBe('one');
    expect(headers.at(1).text()).toBe('two');
    expect(headers.at(2).text()).toBe('three');

    const rowOne = rows.at(1).children('td');
    expect(rowOne.at(0).text()).toBe('1');
    expect(rowOne.at(1).text()).toBe('2');
    expect(rowOne.at(2).text()).toBe('3');

    const rowTwo = rows.at(2).children('td');
    expect(rowTwo.at(0).text()).toBe('4');
    expect(rowTwo.at(1).text()).toBe('5');
    expect(rowTwo.at(2).text()).toBe('6');
  });

  test('should render styles', () => {
    const props = {
      ...minimalProps,
      styles: {
        table: {
          width: 'auto',
          minWidth: '400px',
        },
      },
    };

    wrapper = shallow(<HtmlTableView {...props} />);
    const tableStlye = wrapper.find(Table).get(0).props.style;
    expect(tableStlye).toHaveProperty('width', 'auto');
    expect(tableStlye).toHaveProperty('minWidth', '400px');
  });
});

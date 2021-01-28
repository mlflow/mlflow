import React from 'react';
import { shallow } from 'enzyme';
import { EditableTable } from './EditableFormTable';

describe('unit tests', () => {
  let wrapper;
  const minimalProps = {
    columns: [
      {
        title: 'Name',
        dataIndex: 'name',
        width: 200,
      },
      {
        title: 'Value',
        dataIndex: 'value',
        width: 200,
        editable: true,
      },
    ],
    data: [
      { key: 'tag1', name: 'tag1', value: 'value1' },
      { key: 'tag2', name: 'tag2', value: 'value2' },
    ],
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c) => c) },
    onSaveEdit: () => {},
    onDelete: () => {},
  };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<EditableTable {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

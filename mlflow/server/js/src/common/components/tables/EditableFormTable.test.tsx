/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { EditableTable } from './EditableFormTable';

describe('unit tests', () => {
  let wrapper: any;
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
    onSaveEdit: () => {},
    onDelete: () => {},
  };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<EditableTable {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should display only one modal when deleting a tag', () => {
    // Prep
    wrapper = shallow(<EditableTable {...minimalProps} />);
    const getModal = () => wrapper.find('[data-testid="editable-form-table-remove-modal"]');

    // Assert
    expect(getModal().props().visible).toBeFalsy();

    // Update
    wrapper.setState((state: any) => ({
      ...state,
      deletingKey: 'tag1',
    }));

    // Assert
    expect(getModal().props().visible).toBeTruthy();
    expect(getModal().length).toBe(1);
  });
});

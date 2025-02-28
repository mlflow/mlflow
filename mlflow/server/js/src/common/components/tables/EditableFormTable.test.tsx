import React from 'react';
import { EditableTable } from './EditableFormTable';
import { renderWithIntl, screen } from '../../utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

describe('unit tests', () => {
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
    renderWithIntl(<EditableTable {...minimalProps} />);
    expect(screen.getByText('tag1')).toBeInTheDocument();
  });

  test('should display only one modal when deleting a tag', async () => {
    // Prep
    renderWithIntl(<EditableTable {...minimalProps} />);

    // Assert
    expect(screen.queryByTestId('editable-form-table-remove-modal')).not.toBeInTheDocument();

    // Update
    await userEvent.click(screen.getAllByTestId('editable-table-button-delete')[0]);

    // Assert
    expect(screen.getByTestId('editable-form-table-remove-modal')).toBeVisible();
  });
});

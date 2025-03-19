/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { EditableTagsTableView } from './EditableTagsTableView';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from '../utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';

const editableTableDataTestId = 'editable-table';
const tagNameInputDataTestId = 'tags-form-input-name';
const addTagButtonDataTestId = 'add-tag-button';

describe('unit tests', () => {
  const minimalProps = {
    tags: {
      tag1: { key: 'tag1', value: 'value1' },
      tag2: { key: 'tag2', value: 'value2' },
    },
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c: any) => c) },
    handleAddTag: () => {},
    handleSaveEdit: () => {},
    handleDeleteTag: () => {},
    isRequestPending: false,
  };

  const renderTestComponent = () =>
    renderWithIntl(
      <DesignSystemProvider>
        <BrowserRouter>
          <EditableTagsTableView {...minimalProps} />
        </BrowserRouter>
      </DesignSystemProvider>,
    );

  test('should render with minimal props without exploding', () => {
    renderTestComponent();
    expect(screen.getByTestId(editableTableDataTestId)).toBeInTheDocument();
  });

  test('should validate tag name properly', async () => {
    renderTestComponent();

    await userEvent.type(screen.getByTestId(tagNameInputDataTestId), 'tag1');
    await userEvent.click(screen.getByTestId(addTagButtonDataTestId));
    const validateText = await screen.findByText('Tag "tag1" already exists.');
    expect(validateText).toBeInTheDocument();
  });
});

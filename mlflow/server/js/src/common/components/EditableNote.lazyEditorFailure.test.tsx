import { jest, describe, test, expect } from '@jest/globals';
import React from 'react';
import { EditableNote } from './EditableNote';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

jest.mock('./ThemeAwareReactMde', () => {
  throw new Error('Loading chunk failed');
});

describe('EditableNote lazy editor failure', () => {
  test('keeps the failure inside the editor area', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});

    renderWithIntl(
      <DesignSystemProvider>
        <EditableNote onSubmit={jest.fn(() => Promise.resolve({}))} onCancel={jest.fn()} showEditor />
      </DesignSystemProvider>,
    );

    expect(await screen.findByText(/Something went wrong with this section/)).toBeVisible();
    expect(screen.getByTestId('note-view-outer-container')).toBeInTheDocument();
    expect(screen.getByTestId('editable-note-actions')).toBeInTheDocument();
  });
});

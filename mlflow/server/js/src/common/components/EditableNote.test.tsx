import React from 'react';
import { EditableNote, EditableNoteImpl } from './EditableNote';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

// Mock the Prompt component here. Otherwise, whenever we try to modify the note view's text
// area in the tests, it failed with the "RPC API is not defined" error.
jest.mock('./Prompt', () => {
  return {
    Prompt: jest.fn(() => <div />),
  };
});

const minimalProps = {
  onSubmit: jest.fn(() => Promise.resolve({})),
  onCancel: jest.fn(() => Promise.resolve({})),
};

const commonProps = { ...minimalProps, showEditor: true };

const textAreaDataTestId = 'text-area';
const saveButtonDataTestId = 'editable-note-save-button';

describe('EditableNote', () => {
  test('should render with minimal props without exploding', () => {
    renderWithIntl(<EditableNote {...minimalProps} />);
    expect(screen.getByTestId('note-view-outer-container')).toBeInTheDocument();
  });

  test('renderActions is called and rendered correctly when showEditor is true', () => {
    renderWithIntl(<EditableNote {...commonProps} />);
    expect(screen.getByTestId('note-view-outer-container')).toBeInTheDocument();
    expect(screen.getByTestId('editable-note-actions')).toBeInTheDocument();
  });

  test('handleSubmitClick with successful onSubmit', async () => {
    renderWithIntl(<EditableNote {...commonProps} />);

    await userEvent.type(screen.getByTestId(textAreaDataTestId), 'test note');
    await userEvent.click(screen.getByTestId(saveButtonDataTestId));

    expect(commonProps.onSubmit).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Failed to submit')).not.toBeInTheDocument();
  });

  test('handleRenameExperiment errors correctly', async () => {
    const mockSubmit = jest.fn(() => Promise.reject());
    const props = {
      onSubmit: mockSubmit,
      onCancel: jest.fn(() => Promise.resolve({})),
      showEditor: true,
    };
    renderWithIntl(<EditableNote {...props} />);

    await userEvent.type(screen.getByTestId(textAreaDataTestId), 'test note');
    await userEvent.click(screen.getByTestId(saveButtonDataTestId));

    expect(mockSubmit).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Failed to submit')).toBeInTheDocument();
  });
  test('updates displayed description when defaultMarkdown changes', () => {
    const { rerender } = renderWithIntl(<EditableNote {...minimalProps} defaultMarkdown="first description" />);
    expect(screen.getByText('first description')).toBeInTheDocument();

    rerender(<EditableNote {...minimalProps} defaultMarkdown="second description" />);
    expect(screen.getByText('second description')).toBeInTheDocument();
  });
});

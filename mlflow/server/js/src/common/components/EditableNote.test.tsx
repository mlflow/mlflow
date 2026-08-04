import { jest, describe, test, expect } from '@jest/globals';
import React from 'react';
import { EditableNote, EditableNoteImpl } from './EditableNote';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Set a React controlled textarea's value reliably by going through the native
// HTMLTextAreaElement value setter and dispatching an `input` event.
function setNativeTextareaValue(textarea: HTMLElement, value: string) {
  const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')!.set!;
  nativeSetter.call(textarea, value);
  fireEvent.input(textarea);
}

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
    renderWithIntl(
      <DesignSystemProvider>
        <EditableNote {...minimalProps} />
      </DesignSystemProvider>,
    );
    expect(screen.getByTestId('note-view-outer-container')).toBeInTheDocument();
  });

  test('renderActions is called and rendered correctly when showEditor is true', () => {
    renderWithIntl(
      <DesignSystemProvider>
        <EditableNote {...commonProps} />
      </DesignSystemProvider>,
    );
    expect(screen.getByTestId('note-view-outer-container')).toBeInTheDocument();
    expect(screen.getByTestId('editable-note-actions')).toBeInTheDocument();
  });

  test('handleSubmitClick with successful onSubmit', async () => {
    renderWithIntl(
      <DesignSystemProvider>
        <EditableNote {...commonProps} />
      </DesignSystemProvider>,
    );

    // Use native setter to avoid character-by-character typing which is flaky
    // under memory pressure when running with the full test suite.
    setNativeTextareaValue(screen.getByTestId(textAreaDataTestId), 'test note');
    await userEvent.click(screen.getByTestId(saveButtonDataTestId));

    // Wait for handleSubmitClick's promise chain to resolve and React to re-render
    await waitFor(() => {
      expect(commonProps.onSubmit).toHaveBeenCalledTimes(1);
      expect(screen.getByTestId(saveButtonDataTestId)).toBeEnabled();
    });

    expect(screen.queryByText('Failed to submit')).not.toBeInTheDocument();
  });

  test('handleRenameExperiment errors correctly', async () => {
    const mockSubmit = jest.fn(() => Promise.reject());
    const props = {
      onSubmit: mockSubmit,
      onCancel: jest.fn(() => Promise.resolve({})),
      showEditor: true,
    };
    renderWithIntl(
      <DesignSystemProvider>
        <EditableNote {...props} />
      </DesignSystemProvider>,
    );

    setNativeTextareaValue(screen.getByTestId(textAreaDataTestId), 'test note');
    await userEvent.click(screen.getByTestId(saveButtonDataTestId));

    await waitFor(() => {
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(screen.getByText('Failed to submit')).toBeInTheDocument();
    });
  });
  test('updates displayed description when defaultMarkdown changes', () => {
    const { rerender } = renderWithIntl(<EditableNote {...minimalProps} defaultMarkdown="first description" />);
    expect(screen.getByText('first description')).toBeInTheDocument();

    rerender(<EditableNote {...minimalProps} defaultMarkdown="second description" />);
    expect(screen.getByText('second description')).toBeInTheDocument();
  });
});

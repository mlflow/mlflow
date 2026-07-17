import { describe, jest, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import type { TagAssignmentModalParams } from './useTagAssignmentModal';
import { useTagAssignmentModal } from './useTagAssignmentModal';
import type { KeyValueEntity } from '../types';
import { DesignSystemProvider } from '@databricks/design-system';
import { fireEvent, waitFor, screen } from '@testing-library/react';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

// Set a React controlled input's value reliably by going through the native
// HTMLInputElement value setter (which React's internal change-tracker hooks
// into) and then dispatching an `input` event so React fires onChange.
function setNativeInputValue(input: HTMLElement, value: string) {
  const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
  nativeSetter.call(input, value);
  fireEvent.input(input);
}

describe('useTagAssignmentModal', () => {
  function renderTestComponent(params: Partial<TagAssignmentModalParams> = {}) {
    function TestComponent() {
      const { TagAssignmentModal, showTagAssignmentModal } = useTagAssignmentModal({
        componentIdPrefix: 'test',
        onSubmit: () => Promise.resolve(),
        ...params,
      });

      return (
        <>
          <button onClick={showTagAssignmentModal}>Open Modal</button>
          {TagAssignmentModal}
        </>
      );
    }

    return renderWithIntl(
      <DesignSystemProvider>
        <TestComponent />
      </DesignSystemProvider>,
    );
  }

  test('should call onSubmit with new tags when adding a tag', async () => {
    const mockOnSubmit = jest.fn<any>(() => Promise.resolve());
    renderTestComponent({ onSubmit: mockOnSubmit });

    await userEvent.click(screen.getByRole('button', { name: 'Open Modal' }));

    // Use native value setter + input event to set values in a single
    // operation. This avoids userEvent.type's character-by-character approach
    // which is flaky under memory pressure when running the full test suite.
    const inputs = await screen.findAllByRole('textbox');
    setNativeInputValue(inputs[0], 'newTag');
    setNativeInputValue(inputs[1], 'newValue');

    // Submit the form
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        [{ key: 'newTag', value: 'newValue' }], // newTags
        [], // deletedTags
      );
    });
  });

  test('should not include edited tags in deletedTags when only the value changes', async () => {
    const initialTags: KeyValueEntity[] = [{ key: 'tagToEdit', value: 'oldValue' }];

    const mockOnSubmit = jest.fn<any>(() => Promise.resolve());
    renderTestComponent({ initialTags, onSubmit: mockOnSubmit });

    await userEvent.click(screen.getByRole('button', { name: 'Open Modal' }));

    // Find the value input and change it
    const inputs = await screen.findAllByRole('textbox');
    const valueInput = inputs[1]; // Second input should be the value

    // Clear and type new value
    await userEvent.clear(valueInput);
    await userEvent.type(valueInput, 'newValue');

    // Submit the form
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        [{ key: 'tagToEdit', value: 'newValue' }], // newTags - should include the edited tag
        [], // deletedTags - should be empty because key is not deleted, just value changed
      );
    });
  });
});

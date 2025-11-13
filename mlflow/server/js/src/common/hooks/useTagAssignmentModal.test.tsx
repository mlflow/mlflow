import { describe, jest, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import type { TagAssignmentModalParams } from './useTagAssignmentModal';
import { useTagAssignmentModal } from './useTagAssignmentModal';
import type { KeyValueEntity } from '../types';
import { DesignSystemProvider } from '@databricks/design-system';
import { waitFor, screen } from '@testing-library/react';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

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

    // Find the first key input and type a tag key
    const keyInputs = screen.getAllByRole('textbox');
    await userEvent.type(keyInputs[0], 'newTag');

    // Find the value input (should be the second textbox)
    await userEvent.type(keyInputs[1], 'newValue');

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
    const inputs = screen.getAllByRole('textbox');
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

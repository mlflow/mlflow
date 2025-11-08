import userEvent from '@testing-library/user-event';

import { useEditKeyValueTagsModal } from './useEditKeyValueTagsModal';
import type { KeyValueEntity } from '../types';
import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
  selectAntdOption,
  renderWithIntl,
} from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('useEditKeyValueTagsModal', () => {
  function renderTestComponent(
    editedEntity: { tags?: KeyValueEntity[] } = {},
    allAvailableTags: string[],
    saveTagsHandler = jest.fn(),
    onSuccess = jest.fn(),
  ) {
    function TestComponent() {
      const { showEditTagsModal, EditTagsModal } = useEditKeyValueTagsModal({
        allAvailableTags,
        saveTagsHandler,
        onSuccess,
      });
      return (
        <>
          <button onClick={() => showEditTagsModal(editedEntity)}>trigger button</button>
          {EditTagsModal}
        </>
      );
    }
    const { rerender } = renderWithIntl(<TestComponent />);
    return { rerender: () => rerender(<TestComponent />) };
  }

  test('it should open and close the creation modal properly', async () => {
    renderTestComponent({ tags: [] }, []);
    // When click on trigger button
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // When click on close
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    // Then modal closed
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  test('should show list of provided tags', async () => {
    renderTestComponent({}, ['tag3', 'tag4']);
    // When click on trigger button
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // Then list of tags shown
    await act(async () => {
      const select = within(screen.getByRole('dialog')).getByRole('combobox');
      fireEvent.mouseDown(select);
    });
    expect(screen.getByRole('option', { name: 'tag3' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'tag4' })).toBeInTheDocument();
  });

  const existingTags = [{ key: 'tag1', value: 'tagvalue1' }] as KeyValueEntity[];
  const existingTaggedEntity = { tags: existingTags };

  test('it should properly add tag with key only', async () => {
    const saveHandlerFn = jest.fn().mockResolvedValue({});

    renderTestComponent(existingTaggedEntity, ['tag1', 'tag2'], saveHandlerFn);

    // When click on trigger button to open modal
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // And fill out the form
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('combobox'));
    await userEvent.paste('newtag', {
      clipboardData: { getData: jest.fn() },
    } as any);
    await userEvent.click(screen.getByText(/Add tag "newtag"/));
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(saveHandlerFn).toHaveBeenCalledWith(existingTaggedEntity, existingTags, [
      { key: 'tag1', value: 'tagvalue1' },
      { key: 'newtag', value: '' },
    ]);
  });

  test('it should properly select already existing tag', async () => {
    const saveHandlerFn = jest.fn().mockResolvedValue({});

    renderTestComponent(existingTaggedEntity, ['tag1', 'tag2'], saveHandlerFn);

    // When click on trigger button to open modal
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // And fill out the form
    await selectAntdOption(screen.getByRole('dialog'), 'tag2');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(saveHandlerFn).toHaveBeenCalledWith(existingTaggedEntity, existingTags, [
      { key: 'tag1', value: 'tagvalue1' },
      { key: 'tag2', value: '' },
    ]);
  });

  test('it should properly add tag with key and value', async () => {
    const saveHandlerFn = jest.fn().mockResolvedValue({});

    renderTestComponent(existingTaggedEntity, ['tag1', 'tag2'], saveHandlerFn);

    // When click on trigger button to open modal
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // And fill out the form
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('combobox'));
    await userEvent.paste('newtag', {
      clipboardData: { getData: jest.fn() },
    } as any);

    await userEvent.click(screen.getByText(/Add tag "newtag"/));

    await userEvent.type(screen.getByLabelText('Value (optional)'), 'newvalue');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(saveHandlerFn).toHaveBeenCalledWith(existingTaggedEntity, existingTags, [
      { key: 'tag1', value: 'tagvalue1' },
      { key: 'newtag', value: 'newvalue' },
    ]);
  });

  test('it should properly display error when saving tags', async () => {
    const saveHandlerFn = jest.fn().mockRejectedValue({ message: 'This is a test exception' });

    renderTestComponent(existingTaggedEntity, ['tag1', 'tag2'], saveHandlerFn);

    // When click on trigger button to open modal
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));
    // Then modal shown with correct header
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    // And fill out the form
    await userEvent.type(within(screen.getByRole('dialog')).getByRole('combobox'), 'newtag', {
      delay: 1,
    });

    await userEvent.click(screen.getByText(/Add tag "newtag"/));
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(screen.getByText(/This is a test exception/)).toBeInTheDocument();
  });
});

import { fireEvent, within } from '@testing-library/react';
import { renderHook } from '@testing-library/react-hooks';
import userEvent from '@testing-library/user-event';
import { Control, useForm } from 'react-hook-form';

import { KeyValueEntity } from '../../experiment-tracking/types';
import { screen, waitFor, act, selectAntdOption } from '@mlflow/mlflow/src/common/utils/TestUtils.react17';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react17';

import { TagKeySelectDropdown } from './TagSelectDropdown';

describe('TagKeySelectDropdown', () => {
  function renderTestComponent(allAvailableTags: string[], control: Control<KeyValueEntity>) {
    return renderWithIntl(<TagKeySelectDropdown allAvailableTags={allAvailableTags} control={control} />);
  }

  test('it should render list of tags', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    const { container } = renderTestComponent(['tag1', 'tag2'], result.current.control);
    await act(async () => {
      fireEvent.mouseDown(within(container).getByRole('combobox'));
    });
    expect(screen.getByRole('option', { name: 'tag1' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'tag2' })).toBeInTheDocument();
  });

  test('it should filter tags by search input', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    await act(async () => {
      userEvent.type(input, 'tag1');
    });
    expect(screen.getByRole('option', { name: 'tag1' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('it should filter tags by search input based on lowercase', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    await act(async () => {
      userEvent.type(input, 'TAG1');
    });
    expect(screen.getByRole('option', { name: 'tag1' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('it should give the chance to add a new tag', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    userEvent.type(input, 'tag_non_existing{enter}');
    await waitFor(() => {
      expect(result.current.getValues().key).toBe('tag_non_existing');
    });
  });

  test('it should not allow to add a new tag with invalid characters', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    userEvent.type(input, 'invalid-tag{enter}');
    await waitFor(() => {
      // Do not add the value
      expect(result.current.getValues().key).toBe(undefined);
    });
  });

  test('it should call handleChange with selected tag', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    const { container } = renderTestComponent(['tag1', 'tag2'], result.current.control);
    await selectAntdOption(container, 'tag1');
    await waitFor(() => {
      expect(result.current.getValues().key).toBe('tag1');
    });
  });

  test('it should insert key as lowercase', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    userEvent.type(input, 'TAG_NON_EXISTING{enter}');
    await waitFor(() => {
      expect(result.current.getValues().key).toBe('tag_non_existing');
    });
  });
});

import { fireEvent, renderHook, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Control } from 'react-hook-form';
import { useForm } from 'react-hook-form';

import type { KeyValueEntity } from '../types';
import { screen, waitFor, act, selectAntdOption } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

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
    await userEvent.type(input, 'tag1');
    expect(screen.getByRole('option', { name: 'tag1' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('it should filter tags by search input based on lowercase', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    await userEvent.type(input, 'TAG1');
    expect(screen.getByRole('option', { name: 'tag1' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('it should give the chance to add a new tag', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    await userEvent.type(input, 'tag_non_existing');
    // user-event v14 does not pass down keyCode, so we need to use fireEvent
    fireEvent.keyDown(input, { keyCode: 13 });
    await waitFor(() => {
      expect(result.current.getValues().key).toBe('tag_non_existing');
    });
  });

  test('it should not allow to add a new tag with invalid characters', async () => {
    const { result } = renderHook(() => useForm<KeyValueEntity>());
    renderTestComponent(['tag1', 'tag2'], result.current.control);
    const input = screen.getByRole('combobox');
    await userEvent.type(input, 'invalid-tag');
    // user-event v14 does not pass down keyCode, so we need to use fireEvent
    fireEvent.keyDown(input, { keyCode: 13 });
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
    await userEvent.type(input, 'TAG_NON_EXISTING');
    // user-event v14 does not pass down keyCode, so we need to use fireEvent
    fireEvent.keyDown(input, { keyCode: 13 });
    await waitFor(() => {
      expect(result.current.getValues().key).toBe('tag_non_existing');
    });
  });
});

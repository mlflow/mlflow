import { describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button, DropdownMenu } from '@databricks/design-system';

import { renderWithProviders } from './test-utils/renderWithProviders';
import {
  ReorderableTraceColumnList,
  type ReorderableTraceColumnListProps,
  type ReorderableTraceColumnOption,
} from './ReorderableTraceColumnList';

const COLUMNS: ReorderableTraceColumnOption[] = [
  {
    id: 'start_time',
    label: 'Time',
    reorderLabel: 'Reorder Time column',
    componentId: 'test.column.start_time',
  },
  {
    id: 'input',
    label: 'Input',
    reorderLabel: 'Reorder Input column',
    componentId: 'test.column.input',
  },
  {
    id: 'trace_id',
    label: 'Trace ID',
    reorderLabel: 'Reorder Trace ID column',
    componentId: 'test.column.trace_id',
  },
  {
    id: 'assessment:relevance',
    label: 'relevance',
    reorderLabel: 'Reorder relevance column',
    componentId: 'test.column.assessment',
  },
];

const renderList = (over: Partial<ReorderableTraceColumnListProps> = {}) =>
  renderWithProviders(
    <DropdownMenu.Root open>
      <DropdownMenu.Trigger asChild>
        <Button componentId="test.columns.trigger">Columns</Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        <ReorderableTraceColumnList
          columns={COLUMNS}
          visibleColumns={['start_time', 'input']}
          onToggleColumn={jest.fn()}
          onReorderColumn={jest.fn()}
          {...over}
        />
      </DropdownMenu.Content>
    </DropdownMenu.Root>,
  );

describe('ReorderableTraceColumnList', () => {
  test('renders a drag handle for every column while preserving checked state', async () => {
    await renderList();

    expect(await screen.findByRole('button', { name: 'Reorder Time column' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reorder Input column' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reorder Trace ID column' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reorder relevance column' })).toBeInTheDocument();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Time' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('menuitemcheckbox', { name: 'Trace ID' })).toHaveAttribute('aria-checked', 'false');
  });

  test('Ctrl+ArrowUp requests a move before the preceding column', async () => {
    const user = userEvent.setup();
    const onReorderColumn = jest.fn<ReorderableTraceColumnListProps['onReorderColumn']>();
    await renderList({ onReorderColumn });

    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Input' }));
    await user.keyboard('{Control>}{ArrowUp}{/Control}');

    expect(onReorderColumn).toHaveBeenCalledWith('input', 'start_time');
  });

  test('Ctrl+ArrowDown requests a move after the following column', async () => {
    const user = userEvent.setup();
    const onReorderColumn = jest.fn<ReorderableTraceColumnListProps['onReorderColumn']>();
    await renderList({ onReorderColumn });

    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'Time' }));
    await user.keyboard('{Control>}{ArrowDown}{/Control}');

    expect(onReorderColumn).toHaveBeenCalledWith('start_time', 'input');
  });

  test('dynamic columns can move across the standard-column boundary', async () => {
    const user = userEvent.setup();
    const onReorderColumn = jest.fn<ReorderableTraceColumnListProps['onReorderColumn']>();
    await renderList({ onReorderColumn });

    await user.click(await screen.findByRole('menuitemcheckbox', { name: 'relevance' }));
    await user.keyboard('{Control>}{ArrowUp}{/Control}');

    expect(onReorderColumn).toHaveBeenCalledWith('assessment:relevance', 'trace_id');
  });
});

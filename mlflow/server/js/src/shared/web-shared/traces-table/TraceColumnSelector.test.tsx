import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { TraceColumnSelector, type ColumnSelectorOption } from './TraceColumnSelector';
import type { TraceColumnId } from './types';

const COLUMNS: ColumnSelectorOption[] = [
  { id: 'start_time', label: 'Time', componentId: 'test.col.start_time' },
  { id: 'input', label: 'Input', componentId: 'test.col.input' },
  { id: 'trace_id', label: 'Trace ID', componentId: 'test.col.trace_id' },
];

const renderSelector = (over: Partial<React.ComponentProps<typeof TraceColumnSelector>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TraceColumnSelector
          columns={COLUMNS}
          visibleColumns={['start_time', 'input']}
          onToggleColumn={jest.fn()}
          onResetToDefaults={jest.fn()}
          {...over}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceColumnSelector', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('trigger surfaces the visible/total count via its tooltip', async () => {
    renderSelector();
    await userEvent.hover(screen.getByRole('button', { name: 'Select visible columns' }));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Columns (2/3)');
  });

  test('renders a checkbox per column reflecting visibility', async () => {
    renderSelector();
    await userEvent.click(screen.getByRole('button', { name: 'Select visible columns' }));
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Time' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('menuitemcheckbox', { name: 'Trace ID' })).toHaveAttribute('aria-checked', 'false');
  });

  test('toggling a column calls onToggleColumn with its id', async () => {
    const onToggleColumn = jest.fn<(id: TraceColumnId) => void>();
    renderSelector({ onToggleColumn });
    await userEvent.click(screen.getByRole('button', { name: 'Select visible columns' }));
    await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Trace ID' }));
    expect(onToggleColumn).toHaveBeenCalledWith('trace_id');
  });

  test('reset calls onResetToDefaults', async () => {
    const onResetToDefaults = jest.fn();
    renderSelector({ onResetToDefaults });
    await userEvent.click(screen.getByRole('button', { name: 'Select visible columns' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Reset to defaults' }));
    expect(onResetToDefaults).toHaveBeenCalled();
  });

  describe('groups', () => {
    const onToggle = jest.fn<(id: string) => void>();
    const groups = [
      {
        label: 'Assessments',
        options: [
          { id: 'assessment:relevance', label: 'relevance', componentId: 'test.group.item' },
          { id: 'assessment:safety', label: 'safety', componentId: 'test.group.item' },
        ],
        visibleIds: ['assessment:relevance'],
        onToggle,
      },
    ];

    test('renders the group label and a checkbox per group option reflecting visibility', async () => {
      renderSelector({ groups });
      await userEvent.click(screen.getByRole('button', { name: 'Select visible columns' }));
      expect(await screen.findByText('Assessments')).toBeInTheDocument();
      expect(screen.getByRole('menuitemcheckbox', { name: 'relevance' })).toHaveAttribute('aria-checked', 'true');
      expect(screen.getByRole('menuitemcheckbox', { name: 'safety' })).toHaveAttribute('aria-checked', 'false');
    });

    test('toggling a group option calls its onToggle with the option id', async () => {
      renderSelector({ groups });
      await userEvent.click(screen.getByRole('button', { name: 'Select visible columns' }));
      await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'safety' }));
      expect(onToggle).toHaveBeenCalledWith('assessment:safety');
    });
  });
});

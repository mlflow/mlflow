import { describe, it, expect, jest } from '@jest/globals';
import { useState } from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MetricsFilter } from './MetricsFilter';
import type { MetricFilter, MetricFilterColumn, MetricFilterColumnOption } from './MetricsFilter.utils';

const TEST_COLUMN_OPTIONS: MetricFilterColumnOption[] = [{ value: 'user' as MetricFilterColumn, label: 'User' }];

const renderFilter = (filters: MetricFilter[] = [], setFilters = jest.fn()) =>
  renderWithDesignSystem(
    <MetricsFilter filters={filters} setFilters={setFilters} columnOptions={TEST_COLUMN_OPTIONS} />,
  );

describe('MetricsFilter', () => {
  it('renders the filter button', () => {
    renderFilter();
    expect(screen.getByRole('button', { name: /filters/i })).toBeInTheDocument();
  });

  it('shows the active filter count in the button label', () => {
    const activeFilters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'user', value: 'bob' },
    ];
    renderFilter(activeFilters);
    expect(screen.getByRole('button', { name: /filters \(2\)/i })).toBeInTheDocument();
  });

  it('does not show a count when there are no active filters', () => {
    renderFilter([]);
    expect(screen.getByRole('button', { name: /^filters$/i })).toBeInTheDocument();
  });

  it('calls setFilters with an empty array when the clear button is clicked', async () => {
    const setFilters = jest.fn();
    const activeFilters: MetricFilter[] = [{ column: 'user', value: 'alice' }];
    renderFilter(activeFilters, setFilters);

    await userEvent.click(screen.getByRole('button', { name: /clear filters/i }));
    expect(setFilters).toHaveBeenCalledWith([]);
  });

  it('clears filters via keyboard activation (Enter and Space)', async () => {
    const setFilters = jest.fn();
    const activeFilters: MetricFilter[] = [{ column: 'user', value: 'alice' }];
    renderFilter(activeFilters, setFilters);

    const clearButton = screen.getByRole('button', { name: /clear filters/i });
    clearButton.focus();
    expect(clearButton).toHaveFocus();

    await userEvent.keyboard('{Enter}');
    expect(setFilters).toHaveBeenLastCalledWith([]);

    setFilters.mockClear();
    await userEvent.keyboard(' ');
    expect(setFilters).toHaveBeenLastCalledWith([]);
  });

  describe('filter form (popover)', () => {
    it('opens the filter form when the button is clicked', async () => {
      renderFilter();
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      expect(await screen.findByText('Field')).toBeInTheDocument();
      expect(screen.getByText('Operator')).toBeInTheDocument();
      expect(screen.getByText('Value')).toBeInTheDocument();
    });

    it('shows the column options in the field dropdown', async () => {
      renderFilter();
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');
      expect(screen.getByText('User')).toBeInTheDocument();
    });

    it('adds a new filter row when "Add filter" is clicked', async () => {
      renderFilter();
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');

      const valueInputsBefore = screen.getAllByPlaceholderText('Enter value');
      await userEvent.click(screen.getByRole('button', { name: /add filter/i }));

      const valueInputsAfter = screen.getAllByPlaceholderText('Enter value');
      expect(valueInputsAfter).toHaveLength(valueInputsBefore.length + 1);
    });

    it('removes a filter row when the delete button is clicked', async () => {
      renderFilter();
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');

      await userEvent.click(screen.getByRole('button', { name: /add filter/i }));
      expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(2);

      await userEvent.click(screen.getAllByRole('button', { name: /remove filter/i })[0]);
      expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(1);
    });

    it('resets to one empty row when the last filter row is removed', async () => {
      renderFilter();
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');

      await userEvent.click(screen.getByRole('button', { name: /remove filter/i }));
      expect(screen.getAllByPlaceholderText('Enter value')).toHaveLength(1);
    });

    it('calls setFilters with only complete filters on apply', async () => {
      const setFilters = jest.fn();
      renderFilter([], setFilters);
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');

      const valueInput = screen.getByPlaceholderText('Enter value');
      await userEvent.type(valueInput, 'alice');

      await userEvent.click(screen.getByRole('button', { name: /apply filters/i }));

      await waitFor(() => {
        expect(setFilters).toHaveBeenCalledWith([{ column: 'user', value: 'alice' }]);
      });
    });

    it('calls setFilters with empty array when applying with no values filled in', async () => {
      const setFilters = jest.fn();
      renderFilter([], setFilters);
      await userEvent.click(screen.getByRole('button', { name: /filters/i }));
      await screen.findByText('Field');

      await userEvent.click(screen.getByRole('button', { name: /apply filters/i }));

      expect(setFilters).toHaveBeenCalledWith([]);
    });

    it('initialises the form with existing active filters', async () => {
      const activeFilters: MetricFilter[] = [{ column: 'user', value: 'alice' }];
      renderFilter(activeFilters);
      await userEvent.click(screen.getByRole('button', { name: /^filters/i }));
      await screen.findByText('Field');

      expect(screen.getByDisplayValue('alice')).toBeInTheDocument();
    });

    it('clears the form when filters are cleared externally while the popover is open', async () => {
      const ControlledHarness = () => {
        const [filters, setFilters] = useState<MetricFilter[]>([{ column: 'user', value: 'alice' }]);
        return <MetricsFilter filters={filters} setFilters={setFilters} columnOptions={TEST_COLUMN_OPTIONS} />;
      };
      renderWithDesignSystem(<ControlledHarness />);

      await userEvent.click(screen.getByRole('button', { name: /^filters/i }));
      await screen.findByText('Field');
      expect(screen.getByDisplayValue('alice')).toBeInTheDocument();

      // Click the clear-all control in the trigger; popover stays open due to stopPropagation.
      await userEvent.click(screen.getByRole('button', { name: /clear filters/i }));

      // The form should now reflect the cleared state instead of showing stale rows.
      await waitFor(() => {
        expect(screen.queryByDisplayValue('alice')).not.toBeInTheDocument();
      });
      expect(screen.getByPlaceholderText('Enter value')).toHaveValue('');
    });
  });
});

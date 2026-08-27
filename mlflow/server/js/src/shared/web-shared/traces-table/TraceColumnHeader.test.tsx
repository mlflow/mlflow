import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TraceColumnHeader } from './TraceColumnHeader';
import { renderWithProviders } from './test-utils/renderWithProviders';

const renderHeader = (over: Partial<React.ComponentProps<typeof TraceColumnHeader>> = {}) =>
  renderWithProviders(
    <TraceColumnHeader
      label="Input"
      labelText="Input"
      sortable={false}
      sortDirection="none"
      onSortAscending={jest.fn()}
      onSortDescending={jest.fn()}
      onHide={jest.fn()}
      {...over}
    />,
  );

describe('TraceColumnHeader', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('a non-sortable column offers only Hide column', async () => {
    const onHide = jest.fn();
    renderHeader({ sortable: false, onHide });
    await userEvent.click(screen.getByRole('button', { name: 'Column options for Input' }));

    expect(screen.getByText('Hide column')).toBeInTheDocument();
    expect(screen.queryByText('Sort ascending')).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('Hide column'));
    expect(onHide).toHaveBeenCalledTimes(1);
  });

  test('a sortable column offers Sort ascending/descending plus Hide, wired to the callbacks', async () => {
    const onSortAscending = jest.fn();
    const onSortDescending = jest.fn();
    renderHeader({ sortable: true, sortDirection: 'none', onSortAscending, onSortDescending });
    await userEvent.click(screen.getByRole('button', { name: 'Column options for Input' }));

    await userEvent.click(screen.getByText('Sort ascending'));
    expect(onSortAscending).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByRole('button', { name: 'Column options for Input' }));
    await userEvent.click(screen.getByText('Sort descending'));
    expect(onSortDescending).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByRole('button', { name: 'Column options for Input' }));
    expect(screen.getByText('Hide column')).toBeInTheDocument();
  });
});

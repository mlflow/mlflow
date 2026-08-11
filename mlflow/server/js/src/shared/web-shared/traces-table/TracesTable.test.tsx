import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TRACE_COLUMN_IDS } from './constants';
import type { TraceColumnId } from './types';
import { makeTrace, makeSessionTrace, makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

const ALL_COLUMNS: TraceColumnId[] = [...TRACE_COLUMN_IDS];

const baseProps = (over: Partial<TracesTableProps> = {}): TracesTableProps => ({
  traces: makeTraces(3),
  visibleColumns: ALL_COLUMNS,
  initialColumnSizing: {},
  onColumnSizingSettled: jest.fn(),
  isLoading: false,
  isFetching: false,
  skeletonRowCount: 25,
  onTraceSelected: jest.fn(),
  selectedForBulk: new Map(),
  isAllOnPageSelected: false,
  isSomeOnPageSelected: false,
  onToggleBulkRow: jest.fn(),
  onToggleBulkAll: jest.fn(),
  sort: 'start_time',
  dir: 'desc',
  onSort: jest.fn(),
  ...over,
});

describe('TracesTable', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders a header for each visible column', async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    expect(screen.getByRole('columnheader', { name: /Time/ })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Input/ })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Duration/ })).toBeInTheDocument();
  });

  test('hides columns absent from visibleColumns', async () => {
    await renderWithProviders(<TracesTable {...baseProps({ visibleColumns: ['start_time', 'input'] })} />);
    expect(screen.getByRole('columnheader', { name: /Time/ })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /Duration/ })).not.toBeInTheDocument();
  });

  test('renders skeleton rows (not real rows) while loading', async () => {
    await renderWithProviders(<TracesTable {...baseProps({ isLoading: true, skeletonRowCount: 5 })} />);
    // Real row content (the input preview text) is absent during the skeleton.
    expect(screen.queryByText('request for tr-000')).not.toBeInTheDocument();
  });

  test('clicking a row calls onTraceSelected with that trace', async () => {
    const onTraceSelected = jest.fn();
    const traces = [makeTrace('only')];
    await renderWithProviders(<TracesTable {...baseProps({ traces, onTraceSelected })} />);
    await userEvent.click(screen.getByText('request for only'));
    expect(onTraceSelected).toHaveBeenCalledWith(traces[0]);
  });

  test('only start_time and duration headers expose a sort affordance', async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    // Sortable headers render an aria-sort attribute; non-sortable ones don't.
    expect(screen.getByRole('columnheader', { name: /Time/ })).toHaveAttribute('aria-sort');
    expect(screen.getByRole('columnheader', { name: /Duration/ })).toHaveAttribute('aria-sort');
    expect(screen.getByRole('columnheader', { name: /Input/ })).not.toHaveAttribute('aria-sort');
  });

  test('clicking a sortable header toggles its direction via onSort', async () => {
    const onSort = jest.fn();
    await renderWithProviders(<TracesTable {...baseProps({ sort: 'start_time', dir: 'desc', onSort })} />);
    // A sortable header renders an inner role="button" carrying the toggle handler.
    await userEvent.click(screen.getByRole('button', { name: /Time/ }));
    // start_time is already the active desc sort → toggles to asc.
    expect(onSort).toHaveBeenCalledWith('start_time', 'asc');
  });

  test('the select-all checkbox is indeterminate when some (not all) rows are selected', async () => {
    await renderWithProviders(
      <TracesTable {...baseProps({ isSomeOnPageSelected: true, isAllOnPageSelected: false })} />,
    );
    const selectAll = screen.getByRole('checkbox', { name: /Select all traces on this page/ });
    expect(selectAll).toHaveAttribute('aria-checked', 'mixed');
  });

  test('session cell renders a link when getSessionHref returns a target', async () => {
    const traces = [makeSessionTrace('s1', 'my-session')];
    await renderWithProviders(
      <TracesTable
        {...baseProps({ traces, visibleColumns: ['session'], getSessionHref: () => '/sessions/my-session' })}
      />,
    );
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', expect.stringContaining('/sessions/my-session'));
    expect(within(link).getByText('my-session')).toBeInTheDocument();
  });

  test('session cell renders plain text (no link) when getSessionHref is absent', async () => {
    const traces = [makeSessionTrace('s1', 'my-session')];
    await renderWithProviders(<TracesTable {...baseProps({ traces, visibleColumns: ['session'] })} />);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('my-session')).toBeInTheDocument();
  });
});

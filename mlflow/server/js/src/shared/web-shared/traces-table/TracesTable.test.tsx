import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TRACE_COLUMN_IDS } from './constants';
import type { TraceColumnId } from './types';
import { makeTrace, makeSessionTrace, makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

// Each header renders a per-column options menu; every trigger shares the generic "Column options"
// label, so scope to the target column's header cell before opening its menu.
const openColumnMenu = async (columnName: RegExp) => {
  const header = screen.getByRole('columnheader', { name: columnName });
  await userEvent.click(within(header).getByRole('button', { name: 'Column options' }));
};

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
  onHideColumn: jest.fn(),
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

  test("a sortable column's options menu offers Sort ascending and Sort descending", async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    await openColumnMenu(/Time/); // start_time is server-sortable
    expect(screen.getByText('Sort ascending')).toBeInTheDocument();
    expect(screen.getByText('Sort descending')).toBeInTheDocument();
  });

  test("a non-sortable column's options menu offers only Hide column, no sort items", async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    await openColumnMenu(/Input/); // input is not server-sortable
    expect(screen.getByText('Hide column')).toBeInTheDocument();
    expect(screen.queryByText('Sort ascending')).not.toBeInTheDocument();
    expect(screen.queryByText('Sort descending')).not.toBeInTheDocument();
  });

  test.each([
    { item: 'Sort ascending', dir: 'asc' as const },
    { item: 'Sort descending', dir: 'desc' as const },
  ])('clicking "$item" in a column menu calls onSort with that column id and $dir', async ({ item, dir }) => {
    const onSort = jest.fn();
    await renderWithProviders(<TracesTable {...baseProps({ onSort })} />);
    await openColumnMenu(/Duration/); // exercises columnId mapping on a non-default sortable column
    await userEvent.click(screen.getByText(item));
    expect(onSort).toHaveBeenCalledWith('duration', dir);
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

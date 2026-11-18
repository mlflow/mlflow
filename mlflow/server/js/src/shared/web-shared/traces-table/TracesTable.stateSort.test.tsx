import { describe, expect, jest, test } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TRACE_COLUMN_IDS, getSortableTraceColumns, isSortableTraceColumn } from './constants';
import { makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const props: TracesTableProps = {
  traces: makeTraces(3),
  visibleColumns: [...TRACE_COLUMN_IDS],
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
};

const openColumnMenu = async (columnName: RegExp) => {
  const header = screen.getByRole('columnheader', { name: columnName });
  await userEvent.click(within(header).getByRole('button', { name: 'Column options' }));
};

describe('sorting traces by state', () => {
  test('flag on: state is sortable and its column menu sorts it', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    expect(getSortableTraceColumns()).toEqual(['start_time', 'duration', 'state']);
    expect(isSortableTraceColumn('state')).toBe(true);

    const onSort = jest.fn();
    await renderWithProviders(<TracesTable {...props} onSort={onSort} />);
    await openColumnMenu(/^State/);
    await userEvent.click(screen.getByText('Sort ascending'));
    expect(onSort).toHaveBeenCalledWith('state', 'asc');
  });

  test('flag off: state is not sortable and its menu has no sort items', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    expect(getSortableTraceColumns()).toEqual(['start_time', 'duration']);
    expect(isSortableTraceColumn('state')).toBe(false);

    await renderWithProviders(<TracesTable {...props} />);
    await openColumnMenu(/^State/);
    expect(screen.getByText('Hide column')).toBeInTheDocument();
    expect(screen.queryByText('Sort ascending')).not.toBeInTheDocument();
  });
});

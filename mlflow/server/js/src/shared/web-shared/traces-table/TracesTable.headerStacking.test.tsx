import { describe, expect, jest, test } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TRACE_COLUMN_IDS } from './constants';
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

describe('TracesTable header stacking', () => {
  test('flag on: body rows keep their checkbox below the sticky header', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await renderWithProviders(<TracesTable {...props} />);

    const [headerRow, ...bodyRows] = screen.getAllByRole('row');
    expect(within(headerRow).getByRole('checkbox', { name: 'Select all traces on this page' })).toBeInTheDocument();
    expect(headerRow).toHaveStyle({ zIndex: '2' });
    expect(bodyRows).toHaveLength(3);
    bodyRows.forEach((row) => expect(row.style.isolation).toBe('isolate'));
  });

  test('flag on: the same stacking applies with column reordering', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await renderWithProviders(<TracesTable {...props} onReorderColumn={jest.fn()} />);

    const [headerRow, ...bodyRows] = screen.getAllByRole('row');
    expect(headerRow).toHaveStyle({ zIndex: '2' });
    bodyRows.forEach((row) => expect(row.style.isolation).toBe('isolate'));
  });

  test('flag off: rows keep the default stacking', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    await renderWithProviders(<TracesTable {...props} />);

    screen.getAllByRole('row').forEach((row) => {
      expect(row.style.zIndex).toBe('');
      expect(row.style.isolation).toBe('');
    });
  });
});

import { jest } from '@jest/globals';
import type { TracesTableProps } from '../TracesTable';

/** Ordinary table props the session suites share; each case adds only what its behaviour needs. */
export const makeBaseTableProps = (): Omit<TracesTableProps, 'traces' | 'visibleColumns'> => ({
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
});

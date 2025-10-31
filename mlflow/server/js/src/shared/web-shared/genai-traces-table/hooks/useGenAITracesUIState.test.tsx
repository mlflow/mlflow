import { act, renderHook } from '@testing-library/react';

import { useGenAITracesUIStateColumns, useSelectedColumns } from './useGenAITracesUIState';
import {
  EXECUTION_DURATION_COLUMN_ID,
  INPUTS_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  STATE_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
} from './useTableColumns';
import type { TracesTableColumn } from '../types';
import { TracesTableColumnType } from '../types';

const sort = (a: string[]) => [...a].sort();

const STORAGE_KEY = (expId: string, runUuid?: string) => `genaiTracesUIState-columns-${expId}-${runUuid}`;

const mockColumns = [
  { id: INPUTS_COLUMN_ID, type: TracesTableColumnType.INPUT, label: 'Request' },
  { id: 'col1', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 1' },
  { id: 'col2', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 2' },
  { id: 'col3', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 3' },
  { id: 'col4', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 4' },
  { id: 'col5', type: TracesTableColumnType.ASSESSMENT, label: 'Assessment 5' },
  { id: EXECUTION_DURATION_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Execution Duration' },
  { id: REQUEST_TIME_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Request Time' },
  { id: SOURCE_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Source' },
  { id: TRACE_NAME_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Trace Name' },
  { id: STATE_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'State' },
  { id: 'tags-eval', type: TracesTableColumnType.TRACE_INFO, label: 'Tag - eval' },
];

type UseLSParams = { key: string; initialValue: any };
const memoryStore: Record<string, any> = {};

jest.mock('@databricks/web-shared/hooks', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/hooks')>('@databricks/web-shared/hooks');
  // eslint-disable-next-line @typescript-eslint/no-require-imports, global-require
  const React = require('react');

  return {
    ...actual,
    useLocalStorage: ({ key, initialValue }: UseLSParams) => {
      const [state, setState] = React.useState(key in memoryStore ? memoryStore[key] : initialValue);
      React.useEffect(() => {
        memoryStore[key] = state;
      }, [key, state]);
      return [state, setState] as const;
    },
  };
});

const expId = 'exp-123';

describe('useGenAITracesUIStateColumns', () => {
  beforeEach(() => {
    // eslint-disable-next-line guard-for-in
    for (const k in memoryStore) delete memoryStore[k];
  });

  describe('useSelectedColumns', () => {
    beforeEach(() => {
      // eslint-disable-next-line guard-for-in
      for (const k in memoryStore) delete memoryStore[k];
    });

    it('selectedColumns with all columns -> should max out at 10 columns', () => {
      const { result } = renderHook(() => useSelectedColumns(expId, mockColumns, (cols) => cols));

      expect(sort(result.current.selectedColumns.map((c) => c.id))).toEqual(
        sort([
          INPUTS_COLUMN_ID,
          'col1',
          'col2',
          'col3',
          EXECUTION_DURATION_COLUMN_ID,
          REQUEST_TIME_COLUMN_ID,
          SOURCE_COLUMN_ID,
          STATE_COLUMN_ID,
          TRACE_NAME_COLUMN_ID,
          'tags-eval',
        ]),
      );
    });

    it('setSelectedColumns -> should update selectedColumns', () => {
      const { result } = renderHook(() => useSelectedColumns(expId, mockColumns, (cols) => cols));

      act(() => {
        result.current.setSelectedColumns([
          mockColumns.find((c) => c.id === INPUTS_COLUMN_ID) as TracesTableColumn,
          mockColumns.find((c) => c.id === 'col1') as TracesTableColumn,
          mockColumns.find((c) => c.id === 'col2') as TracesTableColumn,
          mockColumns.find((c) => c.id === 'col3') as TracesTableColumn,
          mockColumns.find((c) => c.id === 'col4') as TracesTableColumn,
        ]);
      });

      expect(sort(result.current.selectedColumns.map((c) => c.id))).toEqual(
        sort([INPUTS_COLUMN_ID, 'col1', 'col2', 'col3', 'col4']),
      );
    });

    it('setSelectedColumns resets all when empty', () => {
      const { result } = renderHook(() => useSelectedColumns(expId, mockColumns, (cols) => cols));

      act(() => {
        result.current.setSelectedColumns([]);
      });

      expect(sort(result.current.selectedColumns.map((c) => c.id))).toEqual(sort([]));
    });
  });

  it('falls back to default hidden columns when storage empty', () => {
    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, mockColumns));

    expect(sort(result.current.hiddenColumns)).toEqual(
      sort([TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID]),
    );
  });

  it('prefers columnOverrides from local storage', () => {
    /* user explicitly hid INPUTS and STATE */
    memoryStore[STORAGE_KEY(expId)] = {
      columnOverrides: {
        [INPUTS_COLUMN_ID]: false,
        [STATE_COLUMN_ID]: false,
        col5: true,
      },
    };

    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, mockColumns));

    expect(sort(result.current.hiddenColumns)).toEqual(
      sort([INPUTS_COLUMN_ID, TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID]),
    );
  });

  it('derives hidden columns from initialSelectedColumns', () => {
    const initialSelected = (cols: typeof mockColumns) => cols.filter((c) => c.id !== 'col1'); // leave col1 un-selected

    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, mockColumns, initialSelected));

    // initialSelected leaves 1 assessment (col1) hidden ➜ 11 visible.
    // The clamp can only show 10, so it hides 1 more assessment (col5).
    // Final hidden set: col1 + col5  = 2 assessments hidden.
    expect(sort(result.current.hiddenColumns)).toEqual(sort(['col1', 'col5']));
  });

  it('derives hidden columns from defaultSelectedColumns and local storage', () => {
    memoryStore[STORAGE_KEY(expId)] = {
      columnOverrides: {
        [INPUTS_COLUMN_ID]: true,
        [REQUEST_TIME_COLUMN_ID]: false,
        [STATE_COLUMN_ID]: false,
        col5: true,
        [TRACE_NAME_COLUMN_ID]: false,
        [SOURCE_COLUMN_ID]: false,
        col2: false,
        col3: true,
      },
    };

    const { result } = renderHook(() =>
      useGenAITracesUIStateColumns(expId, mockColumns, (cols) =>
        cols.filter((c) => c.id !== 'col2' && c.id !== 'col5'),
      ),
    );

    expect(sort(result.current.hiddenColumns)).toEqual(
      sort([REQUEST_TIME_COLUMN_ID, TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, STATE_COLUMN_ID, 'col2']),
    );
  });

  it('toggleColumns updates the in-memory state', () => {
    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, mockColumns));

    /* default hidden first */
    expect(sort(result.current.hiddenColumns)).toEqual(
      sort([TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID]),
    );

    /* toggle INPUTS → should now be hidden too */
    act(() => {
      const inputCol = mockColumns.find((c) => c.id === INPUTS_COLUMN_ID) as TracesTableColumn;
      result.current.toggleColumns([inputCol]);
    });

    expect(sort(result.current.hiddenColumns)).toEqual(
      sort([TRACE_NAME_COLUMN_ID, SOURCE_COLUMN_ID, EXECUTION_DURATION_COLUMN_ID, STATE_COLUMN_ID, INPUTS_COLUMN_ID]),
    );
  });

  it('auto-hides assessment columns to keep ≤10 visible', () => {
    const initialSelected = (cols: typeof mockColumns) => cols; // all visible

    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, mockColumns, initialSelected));

    expect(result.current.hiddenColumns).toEqual(sort(['col4', 'col5'])); // only 3 of 5 remain visible
  });
});

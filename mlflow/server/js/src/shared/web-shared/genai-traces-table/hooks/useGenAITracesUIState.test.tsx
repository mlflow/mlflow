import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useGenAITracesUIStateColumns, useSelectedColumns } from './useGenAITracesUIState';
import {
  EXECUTION_DURATION_COLUMN_ID,
  INPUTS_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  STATE_COLUMN_ID,
  TAGS_COLUMN_ID,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
} from './useTableColumns';
import type { TracesTableColumn } from '../types';
import { TracesTableColumnType } from '../types';

// Mock feature flag - disable URL persistence for existing tests (test old behavior)
jest.mock('../../model-trace-explorer/FeatureUtils', () => ({
  shouldEnableTracesTableStatePersistence: () => false,
}));

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

jest.mock('../../hooks/useLocalStorage', () => {
  const actual = jest.requireActual<typeof import('../../hooks/useLocalStorage')>('../../hooks/useLocalStorage');
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

jest.mock('./useColumnsURL', () => ({
  useColumnsURL: () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports, global-require
    const React = require('react');
    const [urlColumnIds, setUrlColumnIds] = React.useState(undefined);
    return [urlColumnIds, setUrlColumnIds] as const;
  },
}));

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

  it('hides low-priority info columns before hiding assessment columns when over the limit', () => {
    // Simulate a realistic scenario: many info columns (including low-priority ones) + assessment columns.
    // 14 total columns: 7 high-priority + 5 low-priority + 2 assessments.
    // The low-priority columns (run_name, logged_model, user, prompt/linked_prompts, tags) should be hidden
    // before assessment columns are removed.
    const realisticColumns = [
      { id: INPUTS_COLUMN_ID, type: TracesTableColumnType.INPUT, label: 'Request' },
      { id: TRACE_ID_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Trace ID' },
      { id: RESPONSE_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Response' },
      { id: REQUEST_TIME_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Request Time' },
      { id: EXECUTION_DURATION_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Execution Time' },
      { id: STATE_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'State' },
      { id: TOKENS_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Tokens' },
      { id: RUN_NAME_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Run Name' }, // low priority #1
      { id: LOGGED_MODEL_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Version' }, // low priority #2
      { id: USER_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'User' }, // low priority #3
      { id: LINKED_PROMPTS_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Prompt' }, // low priority #4
      { id: TAGS_COLUMN_ID, type: TracesTableColumnType.TRACE_INFO, label: 'Tags' }, // low priority #5
      { id: 'assessment1', type: TracesTableColumnType.ASSESSMENT, label: 'Quality' },
      { id: 'assessment2', type: TracesTableColumnType.ASSESSMENT, label: 'Relevance' },
    ] as TracesTableColumn[];

    // Use defaultSelectedColumns that includes all columns (14 total, needs to trim to 10)
    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, realisticColumns, (cols) => cols));

    // Should hide 4 low-priority info columns, not assessment columns
    const hidden = result.current.hiddenColumns;
    // Assessment columns should NOT be hidden
    expect(hidden).not.toContain('assessment1');
    expect(hidden).not.toContain('assessment2');
    // Exactly 4 columns should be hidden (to bring 14 down to 10)
    expect(hidden.length).toBe(4);
    // All hidden columns must be from the low-priority set
    const lowPriorityIds = [
      RUN_NAME_COLUMN_ID,
      LOGGED_MODEL_COLUMN_ID,
      USER_COLUMN_ID,
      LINKED_PROMPTS_COLUMN_ID,
      TAGS_COLUMN_ID,
    ];
    hidden.forEach((id) => {
      expect(lowPriorityIds).toContain(id);
    });
    // Specifically verify prompt (linked_prompts) and tags are trimmed before assessments
    expect(hidden).toContain(LINKED_PROMPTS_COLUMN_ID);
    expect(hidden).toContain(TAGS_COLUMN_ID);
  });

  it('caps at DEFAULT_MAX_VISIBLE_COLUMNS even when high-priority columns alone exceed limit', () => {
    // 12 high-priority columns with no low-priority or assessment columns
    const manyHighPriorityColumns = Array.from({ length: 12 }, (_, i) => ({
      id: `high_priority_${i}`,
      type: TracesTableColumnType.TRACE_INFO,
      label: `High Priority ${i}`,
    })) as TracesTableColumn[];

    const { result } = renderHook(() => useGenAITracesUIStateColumns(expId, manyHighPriorityColumns, (cols) => cols));

    // Should still be capped at 10
    const visibleCount = manyHighPriorityColumns.length - result.current.hiddenColumns.length;
    expect(visibleCount).toBeLessThanOrEqual(10);
  });
});

import { beforeEach, describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import type { TraceColumnId } from '@databricks/web-shared/traces-table';
import { useTracesV4ColumnOrder } from './useTracesV4ColumnOrder';

const STANDARD_COLUMNS: TraceColumnId[] = ['input', 'output', 'tags'];

describe('useTracesV4ColumnOrder', () => {
  beforeEach(() => window.localStorage.clear());

  test('reorders assessment columns across standard columns and persists the mixed order', () => {
    const renderOrder = () =>
      renderHook(() =>
        useTracesV4ColumnOrder('mixed-order', STANDARD_COLUMNS, ['assessment:relevance', 'assessment:safety']),
      );
    const { result, unmount } = renderOrder();

    act(() => result.current.reorderColumn('assessment:relevance', 'input'));
    expect(result.current.columnOrder.slice(0, 3)).toEqual(['assessment:relevance', 'input', 'output']);

    unmount();
    expect(renderOrder().result.current.columnOrder.slice(0, 3)).toEqual(['assessment:relevance', 'input', 'output']);
  });

  test('appends newly discovered assessments without disturbing the stored order', () => {
    const { result, rerender } = renderHook(
      ({ assessmentIds }: { assessmentIds: string[] }) =>
        useTracesV4ColumnOrder('new-assessment', STANDARD_COLUMNS, assessmentIds),
      { initialProps: { assessmentIds: ['assessment:relevance'] } },
    );
    act(() => result.current.reorderColumn('assessment:relevance', 'output'));

    rerender({
      assessmentIds: ['assessment:relevance', 'assessment:toxicity'],
    });
    expect(result.current.columnOrder).toEqual([
      'input',
      'assessment:relevance',
      'output',
      'tags',
      'assessment:toxicity',
    ]);
  });

  test('retains assessment ids restored from a saved view even when absent from the current page', () => {
    const { result } = renderHook(() => useTracesV4ColumnOrder('saved-order', STANDARD_COLUMNS, []));
    act(() => result.current.setColumnOrder(['assessment:off-page', 'output', 'input']));

    expect(result.current.columnOrder.slice(0, 3)).toEqual(['assessment:off-page', 'output', 'input']);
  });
});

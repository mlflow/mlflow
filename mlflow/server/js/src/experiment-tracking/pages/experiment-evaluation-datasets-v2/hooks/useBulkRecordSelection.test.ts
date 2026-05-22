/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useBulkRecordSelection } from './useBulkRecordSelection';
import { describe } from '@jest/globals';
import { test } from '@jest/globals';
import { expect } from '@jest/globals';

describe('useBulkRecordSelection', () => {
  test('starts empty with all flags false', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b', 'c']));
    expect(result.current.selected.size).toBe(0);
    expect(result.current.isAllVisibleChecked).toBe(false);
    expect(result.current.isSomeVisibleChecked).toBe(false);
  });

  test('toggle adds and removes an id', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b', 'c']));
    act(() => result.current.toggle('a'));
    expect(result.current.selected.has('a')).toBe(true);
    act(() => result.current.toggle('a'));
    expect(result.current.selected.has('a')).toBe(false);
  });

  test('isSomeVisibleChecked reports the partial-selection state', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b', 'c']));
    act(() => result.current.toggle('a'));
    expect(result.current.isSomeVisibleChecked).toBe(true);
    expect(result.current.isAllVisibleChecked).toBe(false);
  });

  test('isAllVisibleChecked flips on once every visible id is selected', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b']));
    act(() => result.current.toggle('a'));
    act(() => result.current.toggle('b'));
    expect(result.current.isAllVisibleChecked).toBe(true);
    expect(result.current.isSomeVisibleChecked).toBe(false);
  });

  test('toggleAll adds every visible id when none-or-some are checked', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b', 'c']));
    act(() => result.current.toggle('a'));
    act(() => result.current.toggleAll());
    expect(result.current.selected.has('a')).toBe(true);
    expect(result.current.selected.has('b')).toBe(true);
    expect(result.current.selected.has('c')).toBe(true);
  });

  test('toggleAll clears every visible id when all are already checked', () => {
    const { result } = renderHook(() => useBulkRecordSelection(['a', 'b']));
    act(() => result.current.toggleAll());
    act(() => result.current.toggleAll());
    expect(result.current.selected.size).toBe(0);
  });

  test('visibility change alone does not prune the selection — callers must clear() explicitly', () => {
    const { result, rerender } = renderHook(({ ids }: { ids: string[] }) => useBulkRecordSelection(ids), {
      initialProps: { ids: ['a', 'b'] },
    });
    act(() => result.current.toggle('a'));
    // The visible ids change (e.g., user paginated or searched) but the selection persists.
    // The consumer is responsible for calling clear() — see DatasetDetailPageContent's effect
    // on [url.search, url.pageIndex]. If this assertion ever flips to expect auto-pruning,
    // re-examine the callers; the safety property they rely on is that clear() empties it.
    rerender({ ids: ['c', 'd'] });
    expect(result.current.selected.has('a')).toBe(true);
  });

  test('clear empties the selection regardless of visibility', () => {
    const { result, rerender } = renderHook(({ ids }: { ids: string[] }) => useBulkRecordSelection(ids), {
      initialProps: { ids: ['a', 'b'] },
    });
    act(() => result.current.toggle('a'));
    rerender({ ids: ['c', 'd'] });
    act(() => result.current.clear());
    expect(result.current.selected.size).toBe(0);
  });
});

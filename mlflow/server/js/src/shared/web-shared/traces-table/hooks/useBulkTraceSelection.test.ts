import { describe, expect, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useBulkTraceSelection } from './useBulkTraceSelection';
import { makeTrace } from '../test-utils/mockTraces';

// Stub traces keyed by id — the selection stores the full ModelTraceInfoV3, not just the id.
const a = makeTrace('a');
const b = makeTrace('b');
const c = makeTrace('c');
const d = makeTrace('d');

const selectedIds = (selected: Map<string, unknown>) => Array.from(selected.keys()).sort();

describe('useBulkTraceSelection', () => {
  test('starts empty with all flags false', () => {
    const { result } = renderHook(() => useBulkTraceSelection([a, b, c]));
    expect(result.current.selected.size).toBe(0);
    expect(result.current.isAllVisibleChecked).toBe(false);
    expect(result.current.isSomeVisibleChecked).toBe(false);
  });

  test('toggle adds then removes a single trace, storing its full info', () => {
    const { result } = renderHook(() => useBulkTraceSelection([a, b]));
    act(() => result.current.toggle(a));
    expect(result.current.selected.has('a')).toBe(true);
    expect(result.current.selected.get('a')).toBe(a);
    expect(result.current.isSomeVisibleChecked).toBe(true);
    expect(result.current.isAllVisibleChecked).toBe(false);
    act(() => result.current.toggle(a));
    expect(result.current.selected.has('a')).toBe(false);
  });

  test('toggleAll selects all visible, then clears them', () => {
    const { result } = renderHook(() => useBulkTraceSelection([a, b, c]));
    act(() => result.current.toggleAll());
    expect(result.current.isAllVisibleChecked).toBe(true);
    act(() => result.current.toggleAll());
    expect(result.current.selected.size).toBe(0);
  });

  test('selection (and its TraceInfo) persists across pages when visibleTraces change', () => {
    const { result, rerender } = renderHook(({ traces }) => useBulkTraceSelection(traces), {
      initialProps: { traces: [a, b] },
    });
    act(() => result.current.toggle(a));
    // Navigate to the next page — different visible traces, but the page-1 selection is retained.
    rerender({ traces: [c, d] });
    expect(result.current.selected.has('a')).toBe(true);
    expect(result.current.selected.get('a')).toBe(a);
    expect(result.current.isAllVisibleChecked).toBe(false);
    act(() => result.current.toggle(c));
    expect(selectedIds(result.current.selected)).toEqual(['a', 'c']);
  });

  test('clear empties the selection', () => {
    const { result } = renderHook(() => useBulkTraceSelection([a, b]));
    act(() => result.current.toggleAll());
    act(() => result.current.clear());
    expect(result.current.selected.size).toBe(0);
  });
});

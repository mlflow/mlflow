import { renderHook, act } from '@testing-library/react';
import { useTraceViewEditMode } from './useTraceViewEditMode';
import type { TraceView, SpanRange } from './useTraceViews';

const makeRange = (label: string, position: number, fromSpanId: string): SpanRange => ({
  from_selector: { span_id: fromSpanId },
  label,
  description: '',
  position,
});

describe('useTraceViewEditMode', () => {
  it('starts in non-edit mode', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    expect(result.current.isEditMode).toBe(false);
    expect(result.current.draftView).toBeNull();
  });

  it('enterEditMode with no argument creates empty draft', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    expect(result.current.isEditMode).toBe(true);
    expect(result.current.draftView).toEqual({
      view_id: '',
      name: '',
      ranges: [],
    });
  });

  it('enterEditMode with existing view clones it', () => {
    const existing: TraceView = {
      view_id: 'tv-1',
      name: 'My View',
      ranges: [makeRange('Range 1', 0, 'span-a')],
    };
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode(existing));
    expect(result.current.draftView?.view_id).toBe('tv-1');
    expect(result.current.draftView?.name).toBe('My View');
    expect(result.current.draftView?.ranges).toHaveLength(1);
  });

  it('exitEditMode clears draft', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.exitEditMode());
    expect(result.current.isEditMode).toBe(false);
    expect(result.current.draftView).toBeNull();
  });

  it('setName updates draft name', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.setName('New Name'));
    expect(result.current.draftView?.name).toBe('New Name');
  });

  it('addRange appends a range with correct position', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.addRange({ span_id: 'span-a' }));
    act(() => result.current.addRange({ span_id: 'span-b' }, { span_id: 'span-c' }));
    expect(result.current.draftView?.ranges).toHaveLength(2);
    expect(result.current.draftView?.ranges[0].from_selector.span_id).toBe('span-a');
    expect(result.current.draftView?.ranges[0].position).toBe(0);
    expect(result.current.draftView?.ranges[1].from_selector.span_id).toBe('span-b');
    expect(result.current.draftView?.ranges[1].to_selector?.span_id).toBe('span-c');
    expect(result.current.draftView?.ranges[1].position).toBe(1);
  });

  it('removeRange removes by index and recalculates positions', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.addRange({ span_id: 'span-a' }));
    act(() => result.current.addRange({ span_id: 'span-b' }));
    act(() => result.current.removeRange(0));
    expect(result.current.draftView?.ranges).toHaveLength(1);
    expect(result.current.draftView?.ranges[0].from_selector.span_id).toBe('span-b');
    expect(result.current.draftView?.ranges[0].position).toBe(0);
  });

  it('updateRange merges partial updates', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.addRange({ span_id: 'span-a' }));
    act(() => result.current.updateRange(0, { label: 'Tool Calls', input_path: '$.query' }));
    expect(result.current.draftView?.ranges[0].label).toBe('Tool Calls');
    expect(result.current.draftView?.ranges[0].input_path).toBe('$.query');
  });

  it('reorderRanges swaps and recalculates positions', () => {
    const { result } = renderHook(() => useTraceViewEditMode());
    act(() => result.current.enterEditMode());
    act(() => result.current.addRange({ span_id: 'span-a' }));
    act(() => result.current.addRange({ span_id: 'span-b' }));
    act(() => result.current.addRange({ span_id: 'span-c' }));
    act(() => result.current.reorderRanges(0, 2));
    expect(result.current.draftView?.ranges[0].from_selector.span_id).toBe('span-b');
    expect(result.current.draftView?.ranges[1].from_selector.span_id).toBe('span-c');
    expect(result.current.draftView?.ranges[2].from_selector.span_id).toBe('span-a');
    expect(result.current.draftView?.ranges.map((r) => r.position)).toEqual([0, 1, 2]);
  });
});

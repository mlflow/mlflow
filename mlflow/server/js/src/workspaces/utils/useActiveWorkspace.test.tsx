import { describe, beforeEach, it, expect } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { getActiveWorkspace, setActiveWorkspace, useActiveWorkspace } from './WorkspaceUtils';

beforeEach(() => {
  // Reset module-level state between tests so subscriber lists and
  // current value start clean.
  setActiveWorkspace(null);
});

describe('useActiveWorkspace', () => {
  it('returns the current active workspace on mount', () => {
    setActiveWorkspace('foo');
    const { result } = renderHook(() => useActiveWorkspace());
    expect(result.current).toBe('foo');
  });

  it('re-renders when setActiveWorkspace is called with a new value', () => {
    const { result } = renderHook(() => useActiveWorkspace());
    expect(result.current).toBe(null);

    act(() => {
      setActiveWorkspace('foo');
    });
    expect(result.current).toBe('foo');

    act(() => {
      setActiveWorkspace('bar');
    });
    expect(result.current).toBe('bar');
  });

  it('clears to null when setActiveWorkspace(null) is called', () => {
    setActiveWorkspace('foo');
    const { result } = renderHook(() => useActiveWorkspace());
    expect(result.current).toBe('foo');

    act(() => {
      setActiveWorkspace(null);
    });
    expect(result.current).toBe(null);
  });

  it('does not notify subscribers when the value is unchanged', () => {
    setActiveWorkspace('foo');
    let renderCount = 0;
    renderHook(() => {
      renderCount += 1;
      return useActiveWorkspace();
    });
    const initialRenders = renderCount;

    act(() => {
      // Setting the same value should be a no-op for subscribers; React
      // would still re-render if the underlying hook reported a change.
      setActiveWorkspace('foo');
    });
    expect(renderCount).toBe(initialRenders);
  });

  it('stays in sync with getActiveWorkspace', () => {
    const { result } = renderHook(() => useActiveWorkspace());
    act(() => {
      setActiveWorkspace('foo');
    });
    expect(result.current).toBe(getActiveWorkspace());
  });
});

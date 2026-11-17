import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { useIsAfterFirstPaint } from './useIsAfterFirstPaint';

describe('useIsAfterFirstPaint', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it('keeps immediate callers immediate', () => {
    const { result } = renderHook(() => useIsAfterFirstPaint(false));
    expect(result.current).toBe(true);
  });

  it('shares one frame and batches subscribers after paint, retaining readiness', () => {
    const frame = jest.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1);
    let renders = 0;
    const { result, rerender } = renderHook(
      ({ defer }) => {
        renders++;
        return [useIsAfterFirstPaint(defer), useIsAfterFirstPaint(defer)];
      },
      { initialProps: { defer: true }, legacyRoot: true },
    );
    expect(result.current).toEqual([false, false]);
    expect(frame).toHaveBeenCalledTimes(1);
    act(() => frame.mock.calls[0][0](0));
    expect(result.current).toEqual([false, false]);
    act(() => jest.runOnlyPendingTimers());
    expect(result.current).toEqual([true, true]);
    expect(renders).toBe(2);
    rerender({ defer: false });
    rerender({ defer: true });
    expect(result.current).toEqual([true, true]);
  });

  it('removes unmounted subscribers before the scheduled paint', () => {
    const frame = jest.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1);
    let renders = 0;
    const { unmount } = renderHook(() => {
      renders++;
      return useIsAfterFirstPaint(true);
    });
    unmount();
    act(() => {
      frame.mock.calls[0][0](0);
      jest.runOnlyPendingTimers();
    });
    expect(renders).toBe(1);
  });
});

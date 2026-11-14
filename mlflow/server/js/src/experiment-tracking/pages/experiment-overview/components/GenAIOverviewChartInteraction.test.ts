import { describe, expect, it } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useGenAIOverviewChartInteraction } from './GenAIOverviewChartInteraction';

describe('useGenAIOverviewChartInteraction', () => {
  it('ignores a null active point', () => {
    const { result } = renderHook(() => useGenAIOverviewChartInteraction(3));

    act(() => {
      result.current.onPointMouseDown(null);
      result.current.onPointMouseMove(null);
      result.current.onPointMouseUp(null);
    });

    expect(result.current.hoveredIndex).toBeUndefined();
    expect(result.current.dragRange).toBeUndefined();
    expect(result.current.selection).toBeUndefined();
  });
});

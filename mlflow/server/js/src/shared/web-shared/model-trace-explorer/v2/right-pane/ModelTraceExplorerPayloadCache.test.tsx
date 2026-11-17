import { describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { ModelTraceExplorerPayloadCacheProvider, useCachedPayloadResult } from './ModelTraceExplorerPayloadCache';

const wrapper = ModelTraceExplorerPayloadCacheProvider;

describe('selected payload cache', () => {
  it('retains recently inspected payloads and evicts the least recently used after eight results', () => {
    const compute = jest.fn((value: number) => ({ value }));
    const { result, rerender } = renderHook(
      ({ value }) => useCachedPayloadResult('test', value, () => compute(value)),
      { initialProps: { value: 0 }, wrapper },
    );
    const first = result.current;
    for (let value = 1; value < 8; value++) rerender({ value });
    rerender({ value: 0 });
    expect(result.current).toBe(first);
    expect(compute).toHaveBeenCalledTimes(8);
    rerender({ value: 8 });
    rerender({ value: 0 });
    expect(result.current).toBe(first);
    expect(compute).toHaveBeenCalledTimes(9);
    rerender({ value: 1 });
    expect(result.current).toEqual({ value: 1 });
    expect(compute).toHaveBeenCalledTimes(10);
  });

  it('separates result kinds and pane lifetimes', () => {
    const payload = { text: 'synthetic' };
    const compute = jest.fn(() => 'result');
    const { rerender } = renderHook(({ kind }) => useCachedPayloadResult(kind, payload, compute), {
      initialProps: { kind: 'json' },
      wrapper,
    });
    rerender({ kind: 'yaml' });
    rerender({ kind: 'json' });
    expect(compute).toHaveBeenCalledTimes(2);
    const otherPane = renderHook(() => useCachedPayloadResult('json', payload, compute), { wrapper });
    expect(otherPane.result.current).toBe('result');
    expect(compute).toHaveBeenCalledTimes(3);
  });
});

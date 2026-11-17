import { describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { ModelTraceExplorerPayloadCacheProvider, useCachedPayloadResult } from './ModelTraceExplorerPayloadCache';

const wrapper = ModelTraceExplorerPayloadCacheProvider;

describe('selected payload cache', () => {
  it('supports consumers outside a pane provider and updates replacement payloads', () => {
    const firstPayload = { text: 'session turn one' };
    const nextPayload = { text: 'session turn two' };
    const compute = jest.fn((payload: { text: string }) => payload.text);
    const { result, rerender } = renderHook(
      ({ payload }) => useCachedPayloadResult('fields', payload, () => compute(payload)),
      { initialProps: { payload: firstPayload } },
    );
    expect(result.current).toBe('session turn one');
    rerender({ payload: firstPayload });
    expect(compute).toHaveBeenCalledTimes(1);
    rerender({ payload: nextPayload });
    expect(result.current).toBe('session turn two');
    expect(compute).toHaveBeenCalledTimes(2);
  });

  it('retains recently inspected payloads and evicts the least recently used after eight results', () => {
    const compute = jest.fn((value: number) => ({ value }));
    const { result, rerender } = renderHook(
      ({ value }) => useCachedPayloadResult('fields', value, () => compute(value)),
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

  it('retains two comparison pairs of fields while JSON results fill their own budget', () => {
    const compute = jest.fn((value: number) => ({ value }));
    const { result, rerender } = renderHook(
      ({ kind, value }) => useCachedPayloadResult(kind, value, () => compute(value)),
      { initialProps: { kind: 'fields' as 'fields' | 'section-json', value: 0 }, wrapper },
    );
    const first = result.current;
    for (let value = 1; value < 8; value++) rerender({ kind: 'fields', value });
    for (let value = 0; value < 8; value++) rerender({ kind: 'section-json', value });
    rerender({ kind: 'fields', value: 0 });
    expect(result.current).toBe(first);
    expect(compute).toHaveBeenCalledTimes(16);
  });

  it('separates result kinds and pane lifetimes', () => {
    const payload = { text: 'synthetic' };
    const compute = jest.fn(() => 'result');
    const { rerender } = renderHook(({ kind }) => useCachedPayloadResult(kind, payload, compute), {
      initialProps: { kind: 'fields' as 'fields' | 'section-json' },
      wrapper,
    });
    rerender({ kind: 'section-json' });
    rerender({ kind: 'fields' });
    expect(compute).toHaveBeenCalledTimes(2);
    const otherPane = renderHook(() => useCachedPayloadResult('fields', payload, compute), { wrapper });
    expect(otherPane.result.current).toBe('result');
    expect(compute).toHaveBeenCalledTimes(3);
  });
});

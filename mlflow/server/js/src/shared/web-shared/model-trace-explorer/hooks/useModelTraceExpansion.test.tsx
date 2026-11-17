import { expect, it } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';

import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { useModelTraceExpansion } from './useModelTraceExpansion';

const nodes = (...keys: string[]) => keys.map((key) => ({ key }) as ModelTraceSpanNode);
const filters = {} as SpanFilterState;

it('reconciles added and removed keys while retaining choices and unchanged Set identity', () => {
  const { result, rerender } = renderHook(
    ({ tree, traceId, query }) => {
      const [expanded, setExpanded] = useState<Set<string | number>>(new Set());
      useModelTraceExpansion(tree, traceId, query, filters, setExpanded);
      return { expanded, setExpanded };
    },
    { initialProps: { tree: nodes('a', 'b'), traceId: 'trace-1', query: '' } },
  );
  act(() => result.current.setExpanded(new Set(['b'])));
  const collapsed = result.current.expanded;
  rerender({ tree: nodes('b', 'a'), traceId: 'trace-1', query: '' });
  expect(result.current.expanded).toBe(collapsed);
  rerender({ tree: nodes('a', 'b', 'c'), traceId: 'trace-1', query: '' });
  expect(result.current.expanded).toEqual(new Set(['b', 'c']));
  rerender({ tree: nodes('a', 'c'), traceId: 'trace-1', query: '' });
  expect(result.current.expanded).toEqual(new Set(['c']));
  rerender({ tree: nodes('a', 'c'), traceId: 'trace-1', query: 'match' });
  expect(result.current.expanded).toEqual(new Set(['a', 'c']));
  act(() => result.current.setExpanded(new Set(['c'])));
  rerender({ tree: nodes('a', 'c'), traceId: 'trace-2', query: 'match' });
  expect(result.current.expanded).toEqual(new Set(['a', 'c']));
});

it('resets expansion when span filter state changes for the same trace and query', () => {
  const tree = nodes('a', 'b');
  const { result, rerender } = renderHook(
    ({ spanFilters }) => {
      const [expanded, setExpanded] = useState<Set<string | number>>(new Set());
      useModelTraceExpansion(tree, 'trace-1', 'match', spanFilters, setExpanded);
      return { expanded, setExpanded };
    },
    { initialProps: { spanFilters: filters } },
  );
  act(() => result.current.setExpanded(new Set(['b'])));
  const collapsed = result.current.expanded;
  rerender({ spanFilters: filters });
  expect(result.current.expanded).toBe(collapsed);
  rerender({ spanFilters: { ...filters } });
  expect(result.current.expanded).toEqual(new Set(['a', 'b']));
});

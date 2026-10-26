import { describe, expect, it } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { FilterOp, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '../../../common/utils/RoutingTestUtils';
import { useEvalRunComparisonTracesUrlState } from './useEvalRunComparisonTracesUrlState';

describe('useEvalRunComparisonTracesUrlState', () => {
  const model: TraceFilterModel = [{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }];
  let lastSearch = '';
  let navigate: ReturnType<typeof useNavigate> = () => {};
  const LocationSpy = () => {
    const search = useLocation().search;
    navigate = useNavigate();
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };
  const renderState = async (query = '') => {
    const result = renderHook(() => useEvalRunComparisonTracesUrlState(), {
      wrapper: ({ children }) => (
        <TestRouter
          initialEntries={[
            `/p?compareRuns=candidate&selectedEvaluationId=trace&q=ordinary&filters=ordinary&sort=cost&dir=asc${query}`,
          ]}
          routes={[
            testRoute(
              <>
                <LocationSpy />
                {children}
              </>,
              '/p',
            ),
          ]}
        />
      ),
    });
    await waitForRoutesToBeRendered();
    return result;
  };
  const preserved = () => {
    const params = new URLSearchParams(lastSearch);
    expect(
      Object.fromEntries(
        ['compareRuns', 'selectedEvaluationId', 'q', 'filters', 'sort', 'dir'].map((key) => [key, params.get(key)]),
      ),
    ).toEqual({
      compareRuns: 'candidate',
      selectedEvaluationId: 'trace',
      q: 'ordinary',
      filters: 'ordinary',
      sort: 'cost',
      dir: 'asc',
    });
  };

  it('reads only the comparison namespace', async () => {
    const { result } = await renderState(
      `&comparisonQ=hello&comparisonSort=duration&comparisonDir=asc&comparisonFilters=${encodeURIComponent(JSON.stringify(model))}`,
    );
    expect(result.current.search).toBe('hello');
    expect(result.current.filterModel).toEqual(model);
    expect(result.current.sort).toBe('duration');
    expect(result.current.dir).toBe('asc');
  });

  it.each(['tokens', 'cost', 'assessment:quality'])('reads comparison sorting by %s', async (column) => {
    const { result } = await renderState(`&comparisonSort=${encodeURIComponent(column)}&comparisonDir=asc`);
    expect(result.current.sort).toBe(column);
    expect(result.current.dir).toBe('asc');
  });

  it('preserves unrelated state when committing and clearing criteria', async () => {
    const { result } = await renderState();
    expect(result.current.search).toBe('');
    expect(result.current.sort).toBe('start_time');
    expect(result.current.dir).toBe('desc');
    act(() => result.current.setSearch('hello'));
    act(() => result.current.setFilterModel(model));
    act(() => result.current.setSort('duration', 'asc'));
    expect(new URLSearchParams(lastSearch).get('comparisonFilters')).toBe(JSON.stringify(model));
    preserved();
    act(() => result.current.clearFilters());
    expect(result.current.search).toBe('hello');
    expect(result.current.filterModel).toEqual([]);
    act(() => result.current.clearAll());
    expect(result.current.search).toBe('');
    expect(result.current.sort).toBe('duration');
    preserved();
  });

  it.each(['bad-json', '{}', '[{"field":"state","operator":"UNKNOWN","value":"OK"}]'])(
    'retains invalid filter state for %s and defaults invalid sorting',
    async (raw) => {
      const { result } = await renderState(
        `&comparisonFilters=${encodeURIComponent(raw)}&comparisonSort=unknown&comparisonDir=asc`,
      );
      expect(result.current.invalidFilters).toBe(true);
      expect(result.current.sort).toBe('start_time');
      expect(result.current.dir).toBe('desc');
      act(() => result.current.clearFilters());
      expect(result.current.invalidFilters).toBe(false);
    },
  );

  it('re-derives criteria when memory-router history changes', async () => {
    const { result } = await renderState('&comparisonQ=first');
    act(() => result.current.setSearch('second'));
    expect(result.current.search).toBe('second');
    act(() => navigate(-1));
    expect(result.current.search).toBe('first');
    act(() => navigate(1));
    expect(result.current.search).toBe('second');
  });

  it.each([
    { query: '', expected: false },
    { query: '&comparisonGroupBy=session', expected: true },
    { query: '&comparisonGroupBy=unsupported&groupBy=session', expected: false },
  ])('reads comparison grouping from $query', async ({ query, expected }) => {
    const { result } = await renderState(query);
    expect(result.current.isGroupedBySession).toBe(expected);
  });

  it.each([true, false])('sets session grouping to %s without replacing other URL state', async (next) => {
    const { result } = await renderState(
      `&comparisonQ=hello&comparisonSort=duration&comparisonDir=asc&comparisonFilters=${encodeURIComponent(JSON.stringify(model))}${next ? '' : '&comparisonGroupBy=session'}`,
    );
    const previousKey = result.current.criteriaKey;
    act(() => result.current.setIsGroupedBySession(next));
    expect(result.current.isGroupedBySession).toBe(next);
    expect(result.current.criteriaKey).not.toBe(previousKey);
    expect(new URLSearchParams(lastSearch).get('comparisonGroupBy')).toBe(next ? 'session' : null);
    expect(result.current.search).toBe('hello');
    expect(result.current.filterModel).toEqual(model);
    expect(result.current.sort).toBe('duration');
    expect(result.current.dir).toBe('asc');
    preserved();
  });

  it('restores the grouping mode after backward and forward navigation', async () => {
    const { result } = await renderState();
    act(() => result.current.setIsGroupedBySession(true));
    act(() => navigate(-1));
    expect(result.current.isGroupedBySession).toBe(false);
    act(() => navigate(1));
    expect(result.current.isGroupedBySession).toBe(true);
    preserved();
  });
});

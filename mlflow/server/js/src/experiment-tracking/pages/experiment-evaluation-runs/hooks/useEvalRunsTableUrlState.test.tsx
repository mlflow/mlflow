import { describe, expect, it } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';

import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import { EVAL_RUNS_SORTING_QUERY_PARAM, useEvalRunsSorting } from './useEvalRunsTableUrlState';

describe('evaluation runs table URL state', () => {
  const mountHook = (initialUrl = '/p', enabled?: boolean) =>
    renderHook(
      () => {
        const [sorting, setSorting] = useEvalRunsSorting(enabled);
        const location = useLocation();
        return { sorting, setSorting, search: location.search };
      },
      {
        wrapper: ({ children }) => (
          <TestRouter initialEntries={[initialUrl]} routes={[testRoute(<div>{children}</div>)]} />
        ),
      },
    );

  const urlWithParams = (params: Record<string, string>) => `/p?${new URLSearchParams(params)}`;
  const getParam = (search: string, key: string) => new URLSearchParams(search).get(key);

  it('uses no sorting when URL state is absent', () => {
    const { result } = mountHook();

    expect(result.current.sorting).toEqual([]);
  });

  it('restores sorting from the URL', () => {
    const sorting = [
      { id: 'metrics.accuracy', desc: true },
      { id: 'attributes.start_time', desc: false },
    ];
    const { result } = mountHook(urlWithParams({ [EVAL_RUNS_SORTING_QUERY_PARAM]: JSON.stringify(sorting) }));

    expect(result.current.sorting).toEqual(sorting);
  });

  it('ignores URL sorting when disabled', () => {
    const sorting = [{ id: 'metrics.accuracy', desc: true }];
    const { result } = mountHook(urlWithParams({ [EVAL_RUNS_SORTING_QUERY_PARAM]: JSON.stringify(sorting) }), false);

    expect(result.current.sorting).toEqual([]);
  });

  it('falls back safely when sorting URL state is malformed', () => {
    const { result } = mountHook(
      urlWithParams({ [EVAL_RUNS_SORTING_QUERY_PARAM]: JSON.stringify([{ id: '', desc: 'yes' }]) }),
    );

    expect(result.current.sorting).toEqual([]);
  });

  it('updates sorting without dropping unrelated URL state', () => {
    const { result } = mountHook('/p?selectedRunUuid=run-1');

    act(() => result.current.setSorting([{ id: 'metrics.accuracy', desc: false }]));
    act(() => result.current.setSorting((current) => current.map((entry) => ({ ...entry, desc: true }))));

    expect(getParam(result.current.search, 'selectedRunUuid')).toBe('run-1');
    expect(JSON.parse(getParam(result.current.search, EVAL_RUNS_SORTING_QUERY_PARAM) ?? '')).toEqual([
      { id: 'metrics.accuracy', desc: true },
    ]);
  });

  it('removes the sorting parameter when sorting is cleared', () => {
    const { result } = mountHook(
      urlWithParams({ [EVAL_RUNS_SORTING_QUERY_PARAM]: JSON.stringify([{ id: 'metrics.accuracy', desc: true }]) }),
    );

    act(() => result.current.setSorting([]));

    expect(getParam(result.current.search, EVAL_RUNS_SORTING_QUERY_PARAM)).toBeNull();
  });
});

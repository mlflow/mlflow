import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import {
  setupTestRouter,
  TestRouter,
  testRoute,
  waitForRoutesToBeRendered,
} from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useTracesV4UrlState } from './useTracesV4UrlState';

describe('useTracesV4UrlState', () => {
  // setupTestRouter registers beforeAll/afterAll hooks; must live at describe scope.
  const { history } = setupTestRouter();

  let lastSearch = '';
  const LocationSpy = () => {
    const search = useLocation().search;
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };

  const mountHook = async (initialUrl: string) => {
    lastSearch = '';
    const result = renderHook(() => useTracesV4UrlState(), {
      wrapper: ({ children }) => (
        <TestRouter
          history={history}
          initialEntries={[initialUrl]}
          routes={[
            testRoute(
              <>
                <LocationSpy />
                <div>{children}</div>
              </>,
            ),
          ]}
        />
      ),
    });
    await waitForRoutesToBeRendered();
    return result;
  };

  const param = (key: string) => new URLSearchParams(lastSearch).get(key);

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('reads search, page, pageSize, sort/dir, and traceId from the URL', async () => {
    const { result } = await mountHook('/p?q=foo&page=3&pageSize=100&sort=duration&dir=asc&traceId=tr-1');
    expect(result.current.search).toBe('foo');
    expect(result.current.pageIndex).toBe(3);
    expect(result.current.pageSize).toBe(100);
    expect(result.current.sort).toBe('duration');
    expect(result.current.dir).toBe('asc');
    expect(result.current.traceId).toBe('tr-1');
  });

  test('defaults: page 1, pageSize 25, start_time DESC, no traceId', async () => {
    const { result } = await mountHook('/p');
    expect(result.current.pageIndex).toBe(1);
    expect(result.current.pageSize).toBe(25);
    expect(result.current.sort).toBe('start_time');
    expect(result.current.dir).toBe('desc');
    expect(result.current.traceId).toBeUndefined();
    expect(result.current.isGroupedBySession).toBe(false);
  });

  test('reads and writes session grouping while resetting pagination', async () => {
    const { result } = await mountHook('/p?page=3&groupBy=session');
    expect(result.current.isGroupedBySession).toBe(true);

    act(() => result.current.setIsGroupedBySession(false));
    expect(param('groupBy')).toBeNull();
    expect(param('page')).toBeNull();

    act(() => result.current.setIsGroupedBySession(true));
    expect(param('groupBy')).toBe('session');
  });

  test('invalid pageSize falls back to the default (25)', async () => {
    const { result } = await mountHook('/p?pageSize=17');
    expect(result.current.pageSize).toBe(25);
  });

  test('a non-server-sortable sort param degrades to the default column', async () => {
    const { result } = await mountHook('/p?sort=tokens&dir=asc');
    // `tokens` is not server-sortable, so it must not be honored.
    expect(result.current.sort).toBe('start_time');
  });

  test('setSearch writes q and clears the page param', async () => {
    const { result } = await mountHook('/p?page=5');
    act(() => result.current.setSearch('hello'));
    expect(param('q')).toBe('hello');
    expect(param('page')).toBeNull();
  });

  test('setPageSize writes pageSize, clears page, and removes the param at the default', async () => {
    const { result } = await mountHook('/p?page=5');
    act(() => result.current.setPageSize(100));
    expect(param('pageSize')).toBe('100');
    expect(param('page')).toBeNull();

    act(() => result.current.setPageSize(25));
    expect(param('pageSize')).toBeNull();
  });

  test('setSort writes params for non-default and clears the page param', async () => {
    const { result } = await mountHook('/p?page=5');
    act(() => result.current.setSort('duration', 'asc'));
    expect(param('sort')).toBe('duration');
    expect(param('dir')).toBe('asc');
    expect(param('page')).toBeNull();
  });

  test('setSort clears params when set back to the default (start_time DESC)', async () => {
    const { result } = await mountHook('/p?sort=duration&dir=asc');
    act(() => result.current.setSort('start_time', 'desc'));
    expect(param('sort')).toBeNull();
    expect(param('dir')).toBeNull();
  });

  test('setTraceId writes and clears the traceId param without touching page', async () => {
    const { result } = await mountHook('/p?page=3');
    act(() => result.current.setTraceId('tr-abc'));
    expect(param('traceId')).toBe('tr-abc');
    expect(param('page')).toBe('3'); // opening the drawer must not reset pagination

    act(() => result.current.setTraceId(undefined));
    expect(param('traceId')).toBeNull();
  });

  test('setPageIndex removes the param when set back to 1', async () => {
    const { result } = await mountHook('/p');
    act(() => result.current.setPageIndex(4));
    expect(param('page')).toBe('4');
    act(() => result.current.setPageIndex(1));
    expect(param('page')).toBeNull();
  });

  describe('tag filters', () => {
    const allTags = () => new URLSearchParams(lastSearch).getAll('tag');

    test('reads repeatable tag params into decoded {key,value} pairs', async () => {
      const { result } = await mountHook('/p?tag=env%3Dprod&tag=team%3Dml');
      expect(result.current.tagFilters).toEqual([
        { key: 'env', value: 'prod' },
        { key: 'team', value: 'ml' },
      ]);
    });

    test('addTagFilter appends an encoded key=value and clears the page param', async () => {
      const { result } = await mountHook('/p?page=5');
      act(() => result.current.addTagFilter('env', 'prod'));
      // getAll reverses URLSearchParams' own outer encoding, so a delimiter-free value keeps its
      // literal `=` separator here (the value only percent-encodes chars that need it).
      expect(allTags()).toEqual(['env=prod']);
      expect(param('page')).toBeNull();
    });

    test('addTagFilter dedupe/toggle: re-adding an identical filter removes it', async () => {
      const { result } = await mountHook('/p?tag=env%3Dprod');
      act(() => result.current.addTagFilter('env', 'prod'));
      expect(allTags()).toEqual([]);
    });

    test('addTagFilter appends distinct filters rather than replacing', async () => {
      const { result } = await mountHook('/p?tag=env%3Dprod');
      act(() => result.current.addTagFilter('team', 'ml'));
      expect(allTags()).toEqual(['env=prod', 'team=ml']);
    });

    test('removeTagFilter drops just the matching filter and clears the page param', async () => {
      const { result } = await mountHook('/p?page=3&tag=env%3Dprod&tag=team%3Dml');
      act(() => result.current.removeTagFilter('env', 'prod'));
      expect(allTags()).toEqual(['team=ml']);
      expect(param('page')).toBeNull();
    });

    test('clearTagFilters removes every tag param', async () => {
      const { result } = await mountHook('/p?tag=env%3Dprod&tag=team%3Dml');
      act(() => result.current.clearTagFilters());
      expect(allTags()).toEqual([]);
    });

    test('round-trips keys/values containing delimiters and special characters', async () => {
      const { result } = await mountHook('/p');
      act(() => result.current.addTagFilter('my.tag key', 'a=b:c d'));
      // The raw param is component-encoded so the delimiters survive…
      expect(allTags()).toEqual(['my.tag%20key=a%3Db%3Ac%20d']);
      // …and decodes back to the exact original key/value.
      expect(result.current.tagFilters).toEqual([{ key: 'my.tag key', value: 'a=b:c d' }]);
    });

    // Regression: `tagFilters` must keep a stable reference across renders when the URL is unchanged.
    // It's rebuilt from `searchParams.getAll('tag')`, and consumers use it as an effect dependency
    // (the controller's clear-selection effect). A fresh array every render re-ran that effect on
    // every render and wiped the bulk selection the instant it was made. Memoizing on the serialized
    // params keeps the identity stable until the tags actually change.
    test('returns a referentially stable tagFilters across re-renders when the URL is unchanged', async () => {
      const { result, rerender } = await mountHook('/p?tag=env%3Dprod');
      const first = result.current.tagFilters;
      rerender();
      expect(result.current.tagFilters).toBe(first);
    });

    test('returns a new tagFilters reference only when the tag params change', async () => {
      const { result } = await mountHook('/p?tag=env%3Dprod');
      const first = result.current.tagFilters;
      act(() => result.current.addTagFilter('team', 'ml'));
      expect(result.current.tagFilters).not.toBe(first);
      expect(result.current.tagFilters).toEqual([
        { key: 'env', value: 'prod' },
        { key: 'team', value: 'ml' },
      ]);
    });
  });
});

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
import { useDatasetRecordsUrlState } from './useDatasetRecordsUrlState';

describe('useDatasetRecordsUrlState', () => {
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
    const result = renderHook(() => useDatasetRecordsUrlState(), {
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

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('reads search, page, and record id from the URL', async () => {
    const { result } = await mountHook('/p?q=foo&page=3&recordId=r-1');

    expect(result.current.search).toBe('foo');
    expect(result.current.pageIndex).toBe(3);
    expect(result.current.recordId).toBe('r-1');
  });

  test('defaults to page 1 and undefined record id when params are absent', async () => {
    const { result } = await mountHook('/p');

    expect(result.current.search).toBe('');
    expect(result.current.pageIndex).toBe(1);
    expect(result.current.recordId).toBeUndefined();
  });

  test('clamps malformed page values up to 1', async () => {
    const { result } = await mountHook('/p?page=-5');

    expect(result.current.pageIndex).toBe(1);
  });

  test('setSearch writes the query param and clears the page param', async () => {
    const { result } = await mountHook('/p?page=5');

    act(() => result.current.setSearch('hello'));

    expect(new URLSearchParams(lastSearch).get('q')).toBe('hello');
    expect(new URLSearchParams(lastSearch).get('page')).toBeNull();
  });

  test('clearing search removes the query param entirely', async () => {
    const { result } = await mountHook('/p?q=old');

    act(() => result.current.setSearch(''));

    expect(new URLSearchParams(lastSearch).get('q')).toBeNull();
  });

  test('setRecordId writes the recordId param; undefined clears it', async () => {
    const { result } = await mountHook('/p?recordId=r-old');

    act(() => result.current.setRecordId('r-new'));
    expect(new URLSearchParams(lastSearch).get('recordId')).toBe('r-new');

    act(() => result.current.setRecordId(undefined));
    expect(new URLSearchParams(lastSearch).get('recordId')).toBeNull();
  });

  test('setPageIndex writes the page param and removes it when set back to 1', async () => {
    const { result } = await mountHook('/p');

    act(() => result.current.setPageIndex(4));
    expect(new URLSearchParams(lastSearch).get('page')).toBe('4');

    act(() => result.current.setPageIndex(1));
    expect(new URLSearchParams(lastSearch).get('page')).toBeNull();
  });

  test('sort and dir default to last_updated DESC when URL params are absent', async () => {
    const { result } = await mountHook('/p');

    expect(result.current.sort).toBe('last_updated');
    expect(result.current.dir).toBe('desc');
  });

  test('reads sort and dir from the URL', async () => {
    const { result } = await mountHook('/p?sort=created_by&dir=asc');

    expect(result.current.sort).toBe('created_by');
    expect(result.current.dir).toBe('asc');
  });

  test('setSort writes sort and dir params for non-default values', async () => {
    const { result } = await mountHook('/p');

    act(() => result.current.setSort('created_by', 'asc'));

    expect(new URLSearchParams(lastSearch).get('sort')).toBe('created_by');
    expect(new URLSearchParams(lastSearch).get('dir')).toBe('asc');
  });

  test('setSort clears params when set back to the default (last_updated DESC)', async () => {
    const { result } = await mountHook('/p?sort=created_by&dir=asc');

    act(() => result.current.setSort('last_updated', 'desc'));

    expect(new URLSearchParams(lastSearch).get('sort')).toBeNull();
    expect(new URLSearchParams(lastSearch).get('dir')).toBeNull();
  });

  test('setSort clears the page param so the user lands on page 1 of the resorted set', async () => {
    const { result } = await mountHook('/p?page=5');

    act(() => result.current.setSort('created_by', 'asc'));

    expect(new URLSearchParams(lastSearch).get('sort')).toBe('created_by');
    expect(new URLSearchParams(lastSearch).get('dir')).toBe('asc');
    expect(new URLSearchParams(lastSearch).get('page')).toBeNull();
  });

  test('malformed dir value falls back to default DESC', async () => {
    const { result } = await mountHook('/p?sort=created_by&dir=banana');

    expect(result.current.sort).toBe('created_by');
    expect(result.current.dir).toBe('desc');
  });
});

import { describe, expect, test } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import {
  setupTestRouter,
  TestRouter,
  testRoute,
  waitForRoutesToBeRendered,
} from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useTracesV4DrawerSessionUrlState } from './useTracesV4DrawerSessionUrlState';

describe('useTracesV4DrawerSessionUrlState', () => {
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
    const result = renderHook(() => useTracesV4DrawerSessionUrlState(), {
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
  const params = () => Object.fromEntries(new URLSearchParams(lastSearch));

  test('reads the session view only when a session id is present', async () => {
    const withSession = await mountHook('/p?traceDrawerView=session&traceDrawerSessionId=s1');
    expect(withSession.result.current).toMatchObject({ sessionId: 's1', viewMode: 'session' });

    const withoutSession = await mountHook('/p?traceDrawerView=session');
    expect(withoutSession.result.current).toMatchObject({ sessionId: undefined, viewMode: 'trace' });
  });

  test('opens a session, shows one of its traces, goes back, and closes, keeping the table parameters', async () => {
    const { result } = await mountHook('/p?groupBy=session&q=hello&spanId=old-span');

    act(() => result.current.openSession('s1', 'tr-1'));
    expect(params()).toEqual({
      groupBy: 'session',
      q: 'hello',
      traceId: 'tr-1',
      traceDrawerView: 'session',
      traceDrawerSessionId: 's1',
    });

    act(() => result.current.showSessionTrace('tr-2'));
    expect(params()).toEqual({ groupBy: 'session', q: 'hello', traceId: 'tr-2', traceDrawerSessionId: 's1' });

    act(() => result.current.showSession());
    expect(params()).toMatchObject({ traceId: 'tr-2', traceDrawerView: 'session', traceDrawerSessionId: 's1' });

    act(() => result.current.closeDrawer());
    expect(params()).toEqual({ groupBy: 'session', q: 'hello' });
  });

  test('selecting another trace leaves the session', async () => {
    const { result } = await mountHook('/p?traceId=tr-1&traceDrawerSessionId=s1');

    act(() => result.current.selectTrace('tr-9'));
    expect(params()).toEqual({ traceId: 'tr-9' });
  });
});

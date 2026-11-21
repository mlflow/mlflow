import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import {
  setupTestRouter,
  TestRouter,
  testRoute,
  waitForRoutesToBeRendered,
} from '../../../common/utils/RoutingTestUtils';
import { useEvalRunTracesV4UrlState } from './useEvalRunTracesV4UrlState';

describe('useEvalRunTracesV4UrlState', () => {
  const { history } = setupTestRouter();
  let lastSearch = '';
  let lastHash = '';
  let lastState: unknown;
  let navigateBack: () => void;

  const LocationSpy = () => {
    const { search, hash, state } = useLocation();
    const navigate = useNavigate();
    navigateBack = () => navigate(-1);
    useEffect(() => {
      lastSearch = search;
      lastHash = hash;
      lastState = state;
    }, [search, hash, state]);
    return null;
  };
  const mountHook = async (
    initialUrl: string | { pathname: string; search: string; hash: string; state: Record<string, boolean> },
  ) => {
    lastSearch = '';
    const result = renderHook(() => useEvalRunTracesV4UrlState(), {
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

  test('prefers the legacy selectedEvaluationId while accepting traceId links', async () => {
    const legacy = await mountHook('/p?traceId=new-link&selectedEvaluationId=legacy-link');
    expect(legacy.result.current.traceId).toBe('legacy-link');

    const v4 = await mountHook('/p?traceId=new-link');
    expect(v4.result.current.traceId).toBe('new-link');
  });

  test('writes the legacy parameter and clears either parameter when the drawer closes', async () => {
    const { result } = await mountHook('/p?page=3&traceId=old-link&spanId=old-span');
    act(() => result.current.setTraceId('trace:/catalog.schema/abc'));
    expect(param('selectedEvaluationId')).toBe('trace:/catalog.schema/abc');
    expect(param('traceId')).toBeNull();
    expect(param('page')).toBe('3');
    expect(param('spanId')).toBeNull();

    act(() => result.current.setTraceId(undefined));
    expect(param('selectedEvaluationId')).toBeNull();
    expect(param('traceId')).toBeNull();
  });

  test('writes the selected span without changing table parameters', async () => {
    const { result } = await mountHook({
      pathname: '/p',
      search: '?page=3&selectedEvaluationId=trace-1&spanId=old-span',
      hash: '#section',
      state: { retained: true },
    });

    act(() => result.current.setSpanId('span-1'));
    expect(param('spanId')).toBe('span-1');
    expect(param('page')).toBe('3');
    expect(lastHash).toBe('#section');
    expect(lastState).toEqual({ retained: true });
    act(() => navigateBack());
    expect(param('spanId')).toBe('span-1');

    act(() => result.current.setSpanId(undefined));
    expect(param('spanId')).toBeNull();
  });
});

import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useLocation } from '../../../common/utils/RoutingUtils';
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

  const LocationSpy = () => {
    const search = useLocation().search;
    useEffect(() => setLastSearch(search), [search]);
    return null;
  };
  const setLastSearch = (search: string) => {
    lastSearch = search;
  };
  const mountHook = async (initialUrl: string) => {
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
});

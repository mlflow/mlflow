import { describe, expect, test } from '@jest/globals';
import { useEffect } from 'react';
import { act, renderHook } from '@testing-library/react';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '../../../../common/utils/RoutingTestUtils';
import { useEvalRunsCompareMode } from './useEvalRunsCompareMode';

describe('useEvalRunsCompareMode', () => {
  const { history } = setupTestRouter();
  let lastSearch = '';

  const LocationSpy = () => {
    const search = useLocation().search;
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };
  const currentParams = () => new URLSearchParams(lastSearch);

  const renderCompareModeHook = (entry: string) =>
    renderHook(() => useEvalRunsCompareMode(), {
      wrapper: ({ children }) => (
        <TestRouter
          history={history}
          initialEntries={[entry]}
          routes={[
            testRoute(
              <>
                <LocationSpy />
                {children}
              </>,
            ),
          ]}
        />
      ),
    });

  test('preserves selected runs while compare mode is toggled off and back on', async () => {
    const { result } = renderCompareModeHook('/?compare=true&compareRuns=run-2,run-3&preserved=value');
    await waitForRoutesToBeRendered();

    expect(result.current.compareMode).toBe(true);
    expect(result.current.compareRuns).toEqual(['run-2', 'run-3']);

    act(() => result.current.setCompareMode(false));

    expect(result.current.compareMode).toBe(false);
    expect(result.current.compareRuns).toEqual(['run-2', 'run-3']);
    expect(currentParams().get('preserved')).toBe('value');

    act(() => result.current.setCompareMode(true));

    expect(result.current.compareMode).toBe(true);
    expect(result.current.compareRuns).toEqual(['run-2', 'run-3']);
  });

  test('does not create a comparison selection when toggling with no selected runs', async () => {
    const { result } = renderCompareModeHook('/?compare=true');
    await waitForRoutesToBeRendered();

    act(() => result.current.setCompareMode(false));
    act(() => result.current.setCompareMode(true));

    expect(result.current.compareMode).toBe(true);
    expect(result.current.compareRuns).toEqual([]);
    expect(currentParams().has('compareRuns')).toBe(false);
  });

  test('keeps list origin when the list comparison switch is toggled off and on', async () => {
    const { result } = renderCompareModeHook('/?compare=true&compareFrom=runs&compareRuns=run-2');
    await waitForRoutesToBeRendered();

    act(() => result.current.setCompareMode(false));
    expect(result.current.fromRunsList).toBe(true);
    expect(result.current.compareMode).toBe(false);

    act(() => result.current.setCompareMode(true));
    expect(result.current.fromRunsList).toBe(true);
    expect(result.current.compareRuns).toEqual(['run-2']);
  });

  test('clears the persisted selection when all comparison runs are removed', async () => {
    const { result } = renderCompareModeHook('/?compare=true&compareRuns=run-2,run-3');
    await waitForRoutesToBeRendered();

    act(() => result.current.setCompareRuns([]));

    expect(result.current.compareMode).toBe(true);
    expect(result.current.compareRuns).toEqual([]);
    expect(currentParams().has('compareRuns')).toBe(false);
  });

  test('starts comparison with a new selection in one URL update', async () => {
    const { result } = renderCompareModeHook('/?compareRuns=old-run&compareFrom=runs&preserved=value');
    await waitForRoutesToBeRendered();

    act(() => result.current.startComparison(['run-2', 'run-3']));

    expect(result.current.compareMode).toBe(true);
    expect(result.current.compareRuns).toEqual(['run-2', 'run-3']);
    const params = currentParams();
    expect(params.get('compare')).toBe('true');
    expect(params.get('compareRuns')).toBe('run-2,run-3');
    expect(params.get('preserved')).toBe('value');
    expect(result.current.fromRunsList).toBe(false);
    expect(params.has('compareFrom')).toBe(false);
  });
});

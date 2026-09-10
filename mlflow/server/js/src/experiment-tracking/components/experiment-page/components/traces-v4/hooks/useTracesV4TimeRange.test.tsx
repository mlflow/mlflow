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
import { useTracesV4TimeRange } from './useTracesV4TimeRange';

const EXPERIMENT_ID = 'exp-1';

describe('useTracesV4TimeRange', () => {
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
    const result = renderHook(() => useTracesV4TimeRange(EXPERIMENT_ID), {
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
    window.localStorage.clear();
  });

  test('reads a relative startTimeLabel from the URL and computes ISO bounds without writing them', async () => {
    const { result } = await mountHook('/p?startTimeLabel=LAST_7_DAYS');
    expect(result.current.timeLabel).toBe('LAST_7_DAYS');
    // A relative label derives its bounds from `dateNow` — they must be resolved…
    expect(result.current.startTime).toBeDefined();
    expect(result.current.endTime).toBeDefined();
    // …but not written to the URL (only CUSTOM carries explicit bounds).
    expect(param('startTime')).toBeNull();
    expect(param('endTime')).toBeNull();
  });

  test('defaults to LAST_7_DAYS when the URL has no label, without writing it to the URL', async () => {
    const { result } = await mountHook('/p');
    expect(result.current.timeLabel).toBe('LAST_7_DAYS');
    expect(param('startTimeLabel')).toBeNull();
  });

  test('an invalid startTimeLabel falls back to the default', async () => {
    const { result } = await mountHook('/p?startTimeLabel=NOT_A_LABEL');
    expect(result.current.timeLabel).toBe('LAST_7_DAYS');
  });

  test('CUSTOM reads explicit start/end bounds from the URL', async () => {
    const start = '2025-01-01T00:00:00.000Z';
    const end = '2025-01-02T00:00:00.000Z';
    const { result } = await mountHook(
      `/p?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(start)}&endTime=${encodeURIComponent(end)}`,
    );
    expect(result.current.timeLabel).toBe('CUSTOM');
    expect(result.current.startTime).toBe(start);
    expect(result.current.endTime).toBe(end);
  });

  test('setTimeRange writes the label and deletes explicit bounds for a non-CUSTOM label', async () => {
    const { result } = await mountHook(
      '/p?startTimeLabel=CUSTOM&startTime=2025-01-01T00%3A00%3A00.000Z&endTime=2025-01-02T00%3A00%3A00.000Z',
    );
    act(() => result.current.setTimeRange({ timeLabel: 'LAST_HOUR' }));
    expect(param('startTimeLabel')).toBe('LAST_HOUR');
    expect(param('startTime')).toBeNull();
    expect(param('endTime')).toBeNull();
  });

  test('setTimeRange writes explicit bounds for CUSTOM', async () => {
    const { result } = await mountHook('/p');
    const start = '2025-03-01T00:00:00.000Z';
    const end = '2025-03-02T00:00:00.000Z';
    act(() => result.current.setTimeRange({ timeLabel: 'CUSTOM', startTime: start, endTime: end }));
    expect(param('startTimeLabel')).toBe('CUSTOM');
    expect(param('startTime')).toBe(start);
    expect(param('endTime')).toBe(end);
  });

  test('timeRangeMs converts the ISO bounds to ms-since-epoch strings', async () => {
    const start = '2025-01-01T00:00:00.000Z';
    const end = '2025-01-02T00:00:00.000Z';
    const { result } = await mountHook(
      `/p?startTimeLabel=CUSTOM&startTime=${encodeURIComponent(start)}&endTime=${encodeURIComponent(end)}`,
    );
    expect(result.current.timeRangeMs.startTime).toBe(String(new Date(start).getTime()));
    expect(result.current.timeRangeMs.endTime).toBe(String(new Date(end).getTime()));
  });

  test('does not read the v3 shared localStorage key (isolated v4 state)', async () => {
    // Seed the legacy v3 key exactly as the shared useMonitoringFilters persists it. v4 must ignore it.
    window.localStorage.setItem(
      `traces_useMonitoringFilters_${EXPERIMENT_ID}`,
      JSON.stringify({ version: 1, data: { startTimeLabel: 'LAST_30_DAYS' } }),
    );
    const { result } = await mountHook('/p');
    // Still the v4 default, not the v3-persisted LAST_30_DAYS.
    expect(result.current.timeLabel).toBe('LAST_7_DAYS');
  });
});

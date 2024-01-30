import { mount } from 'enzyme';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import { getMetricHistoryApiBulk } from '../../../actions';
import { useFetchCompareRunsMetricHistory } from './useFetchCompareRunsMetricHistory';
import { RunInfoEntity } from '../../../types';
import { act } from 'react-dom/test-utils';

jest.mock('../../../actions', () => ({
  getMetricHistoryApiBulk: jest.fn(),
}));

const flushPromises = (advance = false) =>
  new Promise<void>((resolve) => {
    process.nextTick(resolve);
    if (advance) {
      jest.advanceTimersByTime(0);
    }
  });

describe('useFetchCompareRunsMetricHistory', () => {
  const getMetricHistoryApiMock = getMetricHistoryApiBulk as jest.Mock;
  // @ts-expect-error TS(2709): Cannot use namespace 'ReactWrapper' as a type.
  let wrapper: ReactWrapper;

  const isLoadingIndicatorShown = () => wrapper.exists('[data-testid="loading"]');
  const isErrorIndicatorShown = () => wrapper.exists('[data-testid="error"]');

  const mountWrappingComponent = (
    metricKeys: string[],
    runsData: { runInfo: RunInfoEntity }[],
    metricsByRunUuid: any = {},
    enabled = true,
  ) => {
    const MOCK_STATE = {
      entities: {
        metricsByRunUuid,
      },
    };

    const mockStore = configureStore([thunk, promiseMiddleware()]);

    const Component = () => {
      const { isLoading, error } = useFetchCompareRunsMetricHistory(metricKeys, runsData, undefined, enabled);
      return (
        <div>
          {isLoading && <div data-testid="loading" />}
          {error && <div data-testid="error" />}
        </div>
      );
    };

    return mount(
      <Provider store={mockStore(MOCK_STATE)}>
        <Component />
      </Provider>,
    );
  };

  const mockRun = (id: string) => ({ runInfo: { run_uuid: id } } as any);

  beforeEach(() => {
    getMetricHistoryApiMock.mockClear();
    getMetricHistoryApiMock.mockImplementation(() => ({
      type: 'GET_METRIC_HISTORY_API',
      payload: Promise.resolve(),
    }));
  });

  afterEach(() => {
    wrapper.unmount();
  });

  it('fetches metric history for two runs', async () => {
    wrapper = mountWrappingComponent(['metric_1'], [mockRun('run_1'), mockRun('run_2')]);
    expect(getMetricHistoryApiMock).toBeCalledTimes(1);
    expect(getMetricHistoryApiMock).toBeCalledWith(['run_1', 'run_2'], 'metric_1', undefined);

    await flushPromises();
    wrapper.update();

    expect(isLoadingIndicatorShown()).toBe(false);
    expect(isErrorIndicatorShown()).toBe(false);
  });

  it('does not fetch metric history if disabled', async () => {
    wrapper = mountWrappingComponent(['metric_1'], [mockRun('run_1'), mockRun('run_2')], {}, false);
    expect(getMetricHistoryApiMock).toBeCalledTimes(0);

    expect(isLoadingIndicatorShown()).toBe(false);
    expect(isErrorIndicatorShown()).toBe(false);
  });

  it('is not fetching metric history for runs that already have it fetched', async () => {
    const existingMetricHistoryState = {
      run_3: {
        metric_1: [],
      },
    };
    act(() => {
      wrapper = mountWrappingComponent(['metric_1'], [mockRun('run_3'), mockRun('run_4')], existingMetricHistoryState);
    });
    expect(getMetricHistoryApiMock).toBeCalledTimes(1);
    expect(getMetricHistoryApiMock).toBeCalledWith(['run_4'], 'metric_1', undefined);
  });

  it('displays loading indicator', async () => {
    jest.useFakeTimers();
    let fetchPromise: any = null;
    getMetricHistoryApiMock.mockImplementation(() => {
      fetchPromise = new Promise((resolve) => setTimeout(resolve, 2000));
      return {
        type: 'GET_METRIC_HISTORY_API',
        payload: fetchPromise,
      };
    });

    wrapper = mountWrappingComponent(['metric_1'], [mockRun('run_1')], {});
    expect(getMetricHistoryApiMock).toBeCalledTimes(1);
    expect(getMetricHistoryApiMock).toBeCalledWith(['run_1'], 'metric_1', undefined);

    wrapper.update();

    expect(isLoadingIndicatorShown()).toBe(true);

    jest.advanceTimersByTime(3000);
    await fetchPromise;
    await flushPromises(true);
    wrapper.update();

    expect(isLoadingIndicatorShown()).toBe(false);
    jest.useRealTimers();
  });

  it('displays error', async () => {
    getMetricHistoryApiMock.mockImplementation(() => {
      return {
        type: 'GET_METRIC_HISTORY_API_BULK',
        payload: Promise.reject(new Error('some error')),
      };
    });

    wrapper = mountWrappingComponent(['metric_1'], [mockRun('run_1')], {});
    expect(getMetricHistoryApiMock).toBeCalledTimes(1);
    expect(getMetricHistoryApiMock).toBeCalledWith(['run_1'], 'metric_1', undefined);

    await flushPromises(true);
    wrapper.update();

    expect(isErrorIndicatorShown()).toBe(true);
  });
});

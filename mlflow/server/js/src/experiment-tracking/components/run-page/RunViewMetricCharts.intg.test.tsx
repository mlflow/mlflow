import { IntlProvider } from 'react-intl';
import { render, screen, act, within, cleanup, waitFor } from '../../../common/utils/TestUtils.react18';
import { RunViewMetricCharts } from './RunViewMetricCharts';
import type { DeepPartial } from 'redux';
import { applyMiddleware, combineReducers, createStore } from 'redux';
import type { ReduxState } from '../../../redux-types';
import { shouldEnableRunDetailsPageAutoRefresh } from '../../../common/utils/FeatureUtils';
import type { RunsMetricsLinePlotProps } from '../runs-charts/components/RunsMetricsLinePlot';
import LocalStorageUtils from '../../../common/utils/LocalStorageUtils';
import { Provider } from 'react-redux';
import { sampledMetricsByRunUuid } from '../../reducers/SampledMetricsReducer';

import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { latestMetricsByRunUuid, metricsByRunUuid } from '../../reducers/MetricReducer';
import { paramsByRunUuid, tagsByRunUuid } from '../../reducers/Reducers';
import { imagesByRunUuid } from '@mlflow/mlflow/src/experiment-tracking/reducers/ImageReducer';
import { fetchEndpoint } from '../../../common/utils/FetchUtils';
import { DesignSystemProvider } from '@databricks/design-system';

import userEventFactory from '@testing-library/user-event';
import invariant from 'invariant';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../../utils/MetricsUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MlflowService } from '../../sdk/MlflowService';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // increase timeout, it's an integration test with a lot of unmocked code

jest.mock('../runs-charts/components/RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: ({ metricKey }: RunsMetricsLinePlotProps) => {
    return <div data-testid="test-line-plot">Line plot for {metricKey}</div>;
  },
}));

jest.mock('../runs-charts/hooks/useIsInViewport', () => ({
  useIsInViewport: jest.fn(() => ({ isInViewport: true, isInViewportDeferred: true, setElementRef: jest.fn() })),
}));

jest.mock('../../../common/utils/FetchUtils', () => ({
  fetchEndpoint: jest.fn(),
}));

const testRunUuid = 'test_run_uuid';
const testMetricKeys = ['metric_1', 'metric_2', 'system/gpu_1', 'system/gpu_2'];

const testReduxState: DeepPartial<ReduxState> = {
  entities: {
    sampledMetricsByRunUuid: {},
    latestMetricsByRunUuid: {},
    metricsByRunUuid: {},
    paramsByRunUuid: {},
    tagsByRunUuid: {},
    imagesByRunUuid: {},
  },
};

// Exclude setInterval because it's used by waitFor
jest.useFakeTimers({ doNotFake: ['setInterval'] });

const userEvent = userEventFactory.setup({
  advanceTimers: jest.advanceTimersByTime,
});

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableRunDetailsPageAutoRefresh: jest.fn(),
}));

const findChartByTitle = (title: string) => {
  const chartCardElement: HTMLElement | null = screen
    .getByRole('heading', { name: title })
    .closest('[data-testid="experiment-view-compare-runs-card"]');

  invariant(chartCardElement, 'Chart with metric2 should exist');

  return chartCardElement;
};

const getMetricKeyFromEndpointCall = (relativeUrl: string) => {
  const requestParams = new URLSearchParams(relativeUrl);
  return requestParams.get('metric_key');
};
const getLastFetchedMetric = () =>
  getMetricKeyFromEndpointCall(jest.mocked(fetchEndpoint).mock.lastCall?.[0].relativeUrl);

const getLastFetchedMetrics = () =>
  jest.mocked(fetchEndpoint).mock.calls.map((call) => getMetricKeyFromEndpointCall(call[0].relativeUrl));

describe('RunViewMetricCharts - autorefresh', () => {
  const waitForMetricsRequest = () => act(async () => jest.runOnlyPendingTimers());

  beforeEach(() => {
    jest.mocked(shouldEnableRunDetailsPageAutoRefresh).mockImplementation(() => true);
    jest.mocked(fetchEndpoint).mockImplementation(async ({ relativeUrl }) => {
      const requestedKey = getMetricKeyFromEndpointCall(relativeUrl);

      return new Promise((resolve) =>
        setTimeout(
          () =>
            resolve({
              metrics: [{ key: requestedKey, run_id: testRunUuid, step: 1, timestamp: 100, value: 1 }],
            }),
          1000,
        ),
      );
    });

    jest.spyOn(MlflowService, 'listArtifacts').mockImplementation(() => Promise.resolve([]));

    jest.spyOn(LocalStorageUtils, 'getStoreForComponent').mockImplementation(
      () =>
        ({
          setItem: () => ({}),
          getItem: () =>
            JSON.stringify({
              isAccordionReordered: false,
              compareRunCharts: [
                {
                  type: 'LINE',
                  metricSectionId: 'section_id',
                  uuid: 'chart_id',
                  metricKey: 'metric_2',
                  scaleType: 'linear',
                  xAxisKey: 'step',
                  xAxisScaleType: 'linear',
                  range: {
                    xMin: undefined,
                    xMax: undefined,
                    yMin: undefined,
                    yMax: undefined,
                  },
                },
              ],
              compareRunSections: [
                {
                  uuid: 'section_id',
                  name: 'Model metrics',
                  display: true,
                },
              ],
              autoRefreshEnabled: true,
            }),
        } as any),
    );
  });
  const renderComponent = async ({
    mode = 'model',
    metricKeys = testMetricKeys,
    state = testReduxState,
  }: {
    mode?: 'model' | 'system';
    metricKeys?: string[];
    state?: any;
  } = {}) => {
    const runInfo = {
      runUuid: testRunUuid,
    } as any;

    const store = createStore(
      combineReducers({
        entities: combineReducers({
          sampledMetricsByRunUuid,
          latestMetricsByRunUuid,
          metricsByRunUuid,
          paramsByRunUuid,
          tagsByRunUuid,
          imagesByRunUuid,
        }),
      }),
      state,
      applyMiddleware(thunk, promiseMiddleware()),
    );

    await act(async () => {
      render(<RunViewMetricCharts runInfo={runInfo} metricKeys={metricKeys} mode={mode} />, {
        wrapper: ({ children }) => (
          <DesignSystemProvider>
            <TestApolloProvider>
              <Provider store={store}>
                <IntlProvider locale="en">{children}</IntlProvider>
              </Provider>
            </TestApolloProvider>
          </DesignSystemProvider>
        ),
      });
    });
  };

  it('renders a chart for metric_2, adds a new one and auto-refreshes the results', async () => {
    // Render the component and wait for metrics
    await renderComponent({ mode: 'system' });

    // The initial call for metrics should be sent
    expect(fetchEndpoint).toHaveBeenCalledTimes(1);
    expect(getLastFetchedMetric()).toEqual('metric_2');

    // Wait for the metrics to be fetched
    await waitForMetricsRequest();

    // Wait for the auto-refresh interval
    await act(async () => {
      jest.advanceTimersByTime(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL);
    });

    await waitFor(() => {
      // The next call for metrics should be sent
      expect(fetchEndpoint).toHaveBeenCalledTimes(2);
      expect(getLastFetchedMetric()).toEqual('metric_2');
    });

    // Wait for the metrics to be fetched
    await waitForMetricsRequest();

    // Wait for some time (less than full auto refresh interval)
    await act(async () => {
      jest.advanceTimersByTime(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL / 2);
    });

    await waitFor(() => {
      // We should get no new calls
      expect(fetchEndpoint).toHaveBeenCalledTimes(2);
      expect(getLastFetchedMetric()).toEqual('metric_2');
    });

    // Add a new chart. By default, "metric_1" should be selected so we add a chart with "metric_1"
    await userEvent.click(screen.getByRole('button', { name: 'Add chart' }));
    await userEvent.click(screen.getByRole('menuitem', { name: /Line chart/ }));
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Add chart' }));

    await waitFor(() => {
      // We should immediately get a new call
      expect(fetchEndpoint).toHaveBeenCalledTimes(3);
      expect(getLastFetchedMetric()).toEqual('metric_1');
    });

    // Wait for the metrics to be fetched
    await waitForMetricsRequest();

    // Wait for the remainder of the auto-refresh interval
    await act(async () => {
      jest.advanceTimersByTime(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL / 2);
    });

    await waitFor(() => {
      expect(fetchEndpoint).toHaveBeenCalledTimes(4);
      // We should have a call for original metric
      expect(getLastFetchedMetric()).toEqual('metric_2');
    });

    // Wait for the full auto-refresh interval
    await act(async () => {
      jest.advanceTimersByTime(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL);
    });

    await waitFor(() => {
      // We should have two more calls - one for metric_2 and one for metric_1
      expect(fetchEndpoint).toHaveBeenCalledTimes(6);
      expect(getLastFetchedMetrics().slice(-2)).toEqual(expect.arrayContaining(['metric_2', 'metric_1']));
    });

    // Remove "metric_1" chart
    await userEvent.click(within(findChartByTitle('metric_1')).getByTestId('experiment-view-compare-runs-card-menu'));
    await userEvent.click(screen.getByRole('menuitem', { name: /Delete/ }));

    // Wait for the metrics to be fetched after chart is deleted
    await waitForMetricsRequest();

    // Wait for the full auto-refresh interval
    await act(async () => {
      jest.advanceTimersByTime(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL);
    });

    // The next call for "metric_2" should be sent but none for "metric_1"
    await waitFor(() => {
      expect(fetchEndpoint).toHaveBeenCalledTimes(7);
      expect(getLastFetchedMetric()).toEqual('metric_2');
    });

    // Ummount the component
    cleanup();

    // Wait for 10 full auto-refresh intervals
    await act(async () => {
      jest.advanceTimersByTime(10 * EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL);
    });

    // We should get no new calls
    expect(fetchEndpoint).toHaveBeenCalledTimes(7);
  });
});

import { renderWithIntl, act, screen, within } from 'common/utils/TestUtils.react17';
import { RunInfoEntity, SampledMetricsByRunUuidState } from '../../types';
import { RunViewMetricCharts } from './RunViewMetricCharts';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { RunsMetricsLinePlot } from '../runs-charts/components/RunsMetricsLinePlot';
import { RunsMetricsBarPlot } from '../runs-charts/components/RunsMetricsBarPlot';
import userEvent from '@testing-library/user-event';
import { RunsChartsLineChartXAxisType, createChartAxisRangeKey } from '../runs-charts/components/RunsCharts.common';
import { getSampledMetricHistoryBulkAction } from '../../sdk/SampledMetricHistoryService';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { DesignSystemProvider } from '@databricks/design-system';

jest.mock('../runs-charts/components/RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: jest.fn(() => <div />),
}));
jest.mock('../runs-charts/components/RunsMetricsBarPlot', () => ({
  RunsMetricsBarPlot: jest.fn(() => <div />),
}));
jest.mock('../../sdk/SampledMetricHistoryService', () => ({
  getSampledMetricHistoryBulkAction: jest.fn().mockReturnValue({ type: 'getSampledMetricHistoryBulkAction' }),
}));
jest.mock('../../../common/utils/FeatureUtils');

const testRunInfo: RunInfoEntity = {
  experiment_id: 123,
  run_uuid: 'run_1',
  run_name: 'test_run_name',
} as any;

const defaultRangeKey = createChartAxisRangeKey();
const defaultSampledState = {
  run_1: {
    met1: { [defaultRangeKey]: { loading: false, metricsHistory: [] } },
    met2: { [defaultRangeKey]: { loading: false, metricsHistory: [] } },
  },
};

describe('RunViewMetricCharts', () => {
  const renderComponent = ({
    metricKeys = ['met1', 'met2'],
    state = defaultSampledState,
  }: {
    metricKeys?: string[];
    state?: SampledMetricsByRunUuidState;
  }) => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const MOCK_STATE = {
      entities: {
        sampledMetricsByRunUuid: state,
      },
    };
    return renderWithIntl(
      <Provider store={mockStore(MOCK_STATE)}>
        <DesignSystemProvider>
          <RunViewMetricCharts mode="model" metricKeys={metricKeys} runInfo={testRunInfo} />
        </DesignSystemProvider>
      </Provider>,
    );
  };

  beforeEach(() => {
    jest.mocked(RunsMetricsLinePlot).mockClear();
    jest.mocked(RunsMetricsBarPlot).mockClear();
    jest.mocked(getSampledMetricHistoryBulkAction).mockClear();
  });
  test('Displays chart section for generic metrics and filters out metric keys', async () => {
    renderComponent({});
    expect(screen.getByRole('heading', { name: 'met1' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'met2' })).toBeInTheDocument();

    await act(async () => {
      userEvent.paste(screen.getByRole('searchbox'), 'met1');
    });

    expect(screen.queryByRole('heading', { name: 'met1' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'met2' })).not.toBeInTheDocument();
  });

  test('Properly filters out system metric keys', async () => {
    renderComponent({
      metricKeys: ['system/metric_1', 'system/metric_2', 'system/metric_3', 'system/alpha', 'system/beta'],
    });

    ['system/metric_1', 'system/metric_2', 'system/metric_3', 'system/alpha', 'system/beta'].forEach((key) => {
      expect(screen.getByRole('heading', { name: key })).toBeInTheDocument();
    });

    await act(async () => {
      userEvent.paste(screen.getByRole('searchbox'), 'metric_2');
    });

    ['system/metric_1', 'system/metric_3', 'system/alpha', 'system/beta'].forEach((key) => {
      expect(screen.queryByRole('heading', { name: key })).not.toBeInTheDocument();
    });
    expect(screen.queryByRole('heading', { name: 'system/metric_2' })).toBeInTheDocument();

    await act(async () => {
      userEvent.clear(screen.getByRole('searchbox'));
      userEvent.paste(screen.getByRole('searchbox'), 'metric');
    });

    ['system/alpha', 'system/beta'].forEach((key) => {
      expect(screen.queryByRole('heading', { name: key })).not.toBeInTheDocument();
    });
    ['system/metric_1', 'system/metric_2', 'system/metric_3'].forEach((key) => {
      expect(screen.queryByRole('heading', { name: key })).toBeInTheDocument();
    });
  });

  test('Fetches missing data for charts', async () => {
    act(() => {
      renderComponent({
        metricKeys: ['met1', 'system/sysmet1'],
        state: {},
      });
    });

    expect(getSampledMetricHistoryBulkAction).toBeCalledWith(['run_1'], 'met1', expect.anything(), undefined);
    expect(getSampledMetricHistoryBulkAction).toBeCalledWith(['run_1'], 'system/sysmet1', expect.anything(), undefined);
  });

  test('Renders correct amount and types of charts with necessary props and x-axis', async () => {
    const singleValueMetric: any = { step: 1, key: 'met_single', value: 123, timestamp: 1 };
    const modelMetricHistory: any = [
      { step: 1, key: 'met_history_1', value: 123, timestamp: 1 },
      { step: 2, key: 'met_history_1', value: 124, timestamp: 2 },
    ];
    const systemMetricHistory: any = [
      { step: 1, key: 'system/met_history_2', value: 123, timestamp: 100 },
      { step: 2, key: 'system/met_history_2', value: 123, timestamp: 101 },
    ];
    const state: SampledMetricsByRunUuidState = {
      run_1: {
        met_single: {
          [defaultRangeKey]: { metricsHistory: [singleValueMetric], loading: false },
        },
        met_history_1: {
          [defaultRangeKey]: { metricsHistory: modelMetricHistory, loading: false },
        },
        'system/met_history_2': {
          [defaultRangeKey]: { metricsHistory: systemMetricHistory, loading: false },
        },
      },
    };
    renderComponent({
      metricKeys: ['met_single', 'met_history_1', 'system/met_history_2'],
      state,
    });

    // Expect to render a bar plot for a single value metric
    expect(RunsMetricsBarPlot).toHaveBeenLastCalledWith(
      expect.objectContaining({
        metricKey: 'met_single',
        runsData: [
          {
            uuid: 'run_1',
            displayName: 'test_run_name',
            color: expect.anything(),
            metrics: { met_single: singleValueMetric },
            runInfo: testRunInfo,
          },
        ],
      }),
      expect.anything(),
    );

    // Expect to render a line plot for system metric with time-based x-axis
    expect(RunsMetricsLinePlot).toHaveBeenCalledWith(
      expect.objectContaining({
        lineSmoothness: 0,
        metricKey: 'system/met_history_2',
        runsData: [
          {
            uuid: 'run_1',
            displayName: 'test_run_name',
            color: expect.anything(),
            metricsHistory: { 'system/met_history_2': systemMetricHistory },
            runInfo: testRunInfo,
          },
        ],
        xAxisKey: 'time',
      }),
      expect.anything(),
    );

    // Expect to render a line plot for a model metric with step-based x-axis
    expect(RunsMetricsLinePlot).toHaveBeenCalledWith(
      expect.objectContaining({
        lineSmoothness: 0,
        metricKey: 'met_history_1',
        runsData: [
          {
            uuid: 'run_1',
            displayName: 'test_run_name',
            color: expect.anything(),
            metricsHistory: { met_history_1: modelMetricHistory },
            runInfo: testRunInfo,
          },
        ],
        xAxisKey: RunsChartsLineChartXAxisType.STEP,
      }),
      expect.anything(),
    );
  });

  test('Renders loading and error states where necessary', async () => {
    const systemMetricHistory: any = [{ step: 1, key: 'system/met_history_2', value: 123, timestamp: 100 }];
    const state: SampledMetricsByRunUuidState = {
      run_1: {
        metric_valid: {
          [defaultRangeKey]: { metricsHistory: [systemMetricHistory], loading: false },
        },
        metric_loading: {
          [defaultRangeKey]: { metricsHistory: undefined, loading: true },
        },
        metric_error: {
          [defaultRangeKey]: {
            metricsHistory: undefined,
            loading: false,
            error: new ErrorWrapper({ message: 'This is an exception' }),
          },
        },
      },
    };

    renderComponent({
      metricKeys: ['metric_valid', 'metric_loading', 'metric_error'],
      state,
    });

    expect(RunsMetricsBarPlot).toHaveBeenCalledWith(
      expect.not.objectContaining({ metricKey: 'metric_loading' }),
      expect.anything(),
    );
    expect(RunsMetricsBarPlot).toHaveBeenCalledWith(
      expect.not.objectContaining({ metricKey: 'metric_error' }),
      expect.anything(),
    );

    expect(screen.getByText('Error while fetching chart data')).toBeInTheDocument();
    expect(screen.getByText('This is an exception')).toBeInTheDocument();
  });

  test('Changes order of metric charts', async () => {
    renderComponent({
      metricKeys: ['system/sysmet1', 'system/sysmet2', 'system/sysmet3'],
      state: {
        run_1: {
          'system/sysmet1': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
          'system/sysmet2': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
          'system/sysmet3': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
        },
      },
    });

    const firstMetricHeading = screen.getByRole('heading', { name: 'system/sysmet1' });
    const firstChartArea = firstMetricHeading.closest('[role="figure"]') as HTMLElement;

    const dropdownTrigger = within(firstChartArea).getByRole('button', { name: 'Chart options' });

    await act(async () => {
      await openDropdownMenu(dropdownTrigger);
    });

    expect(screen.getByText('Move up')).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByText('Move down')).not.toHaveAttribute('aria-disabled');
    await act(async () => {
      userEvent.click(screen.getByText('Move down'));
    });

    const chartHeadings = screen
      .getAllByRole('figure')
      .map((chartArea) => within(chartArea).getByRole('heading').textContent);

    expect(chartHeadings).toEqual(['system/sysmet2', 'system/sysmet1', 'system/sysmet3']);
  });

  test('Refreshes the data for visible charts', async () => {
    renderComponent({
      metricKeys: ['system/sysmet1', 'system/sysmet2', 'system/sysmet3'],
      state: {
        run_1: {
          'system/sysmet1': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
          'system/sysmet2': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
          'system/sysmet3': {
            [defaultRangeKey]: { loading: false, metricsHistory: [] },
          },
        },
      },
    });

    expect(getSampledMetricHistoryBulkAction).toHaveBeenCalledTimes(3);

    userEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(getSampledMetricHistoryBulkAction).toHaveBeenCalledTimes(6);
  });
});

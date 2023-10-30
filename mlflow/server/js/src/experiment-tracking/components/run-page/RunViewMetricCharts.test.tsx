import { act, renderWithIntl, screen } from '../../../common/utils/TestUtils';
import { MetricHistoryByName, RunInfoEntity } from '../../types';
import { RunViewMetricCharts } from './RunViewMetricCharts';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { RunsMetricsLinePlot } from '../runs-charts/components/RunsMetricsLinePlot';
import { RunsMetricsBarPlot } from '../runs-charts/components/RunsMetricsBarPlot';
import userEvent from '@testing-library/user-event';
import { useFetchCompareRunsMetricHistory } from '../runs-compare/hooks/useFetchCompareRunsMetricHistory';

jest.mock('../runs-charts/components/RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: jest.fn(() => <div />),
}));
jest.mock('../runs-charts/components/RunsMetricsBarPlot', () => ({
  RunsMetricsBarPlot: jest.fn(() => <div />),
}));
jest.mock('../runs-compare/hooks/useFetchCompareRunsMetricHistory', () => ({
  useFetchCompareRunsMetricHistory: jest.fn(() => ({ isLoading: false })),
}));

const testRunInfo: RunInfoEntity = {
  experiment_id: 123,
  run_uuid: 'run_1',
} as any;

describe('RunViewMetricCharts', () => {
  const renderComponent = ({
    metricKeys = ['met1', 'met2'],
    metricHistory = {
      met1: [],
      met2: [],
    },
  }: {
    metricKeys?: string[];
    metricHistory?: MetricHistoryByName;
  }) => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const MOCK_STATE = {
      entities: {
        metricsByRunUuid: {
          run_1: metricHistory,
        },
      },
    };
    renderWithIntl(
      <Provider store={mockStore(MOCK_STATE)}>
        <RunViewMetricCharts metricKeys={metricKeys} runInfo={testRunInfo} />
      </Provider>,
    );
  };

  beforeEach(() => {
    jest.mocked(RunsMetricsLinePlot).mockClear();
    jest.mocked(RunsMetricsBarPlot).mockClear();
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

  test('Displays chart sections for system and model metrics', async () => {
    renderComponent({
      metricKeys: ['met1', 'system/sysmet1', 'system/sysmet2'],
      metricHistory: {
        met1: [],
        'system/sysmet1': [],
        'system/sysmet2': [],
      },
    });
    expect(screen.getByRole('heading', { name: 'met1' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'sysmet1' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'sysmet2' })).toBeInTheDocument();

    expect(screen.getByRole('heading', { name: 'System metrics (2)' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Model metrics (1)' })).toBeInTheDocument();

    await act(async () => {
      userEvent.paste(screen.getByRole('searchbox'), 'sysmet1');
    });

    expect(screen.getByText('No matching metric keys')).toBeInTheDocument();
  });

  test('Fetches missing data for charts', async () => {
    act(() => {
      renderComponent({
        metricKeys: ['met1', 'system/sysmet1'],
        metricHistory: undefined,
      });
    });

    expect(useFetchCompareRunsMetricHistory).toBeCalledWith(
      ['met1', 'system/sysmet1'],
      [{ runInfo: testRunInfo }],
    );
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
    const metricHistory = {
      met_single: [singleValueMetric],
      met_history_1: modelMetricHistory,
      'system/met_history_2': systemMetricHistory,
    };
    renderComponent({
      metricKeys: ['met_single', 'met_history_1', 'system/met_history_2'],
      metricHistory,
    });

    // Expect to render a bar plot for a single value metric
    expect(RunsMetricsBarPlot).toHaveBeenLastCalledWith(
      expect.objectContaining({
        barLabelTextPosition: 'inside',
        metricKey: 'met_single',
        runsData: [
          {
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
            color: expect.anything(),
            metricsHistory: metricHistory,
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
            color: expect.anything(),
            metricsHistory: metricHistory,
            runInfo: testRunInfo,
          },
        ],
        xAxisKey: 'step',
      }),
      expect.anything(),
    );
  });
});

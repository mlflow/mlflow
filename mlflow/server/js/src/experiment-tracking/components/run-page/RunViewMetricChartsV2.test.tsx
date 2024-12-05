import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { render, screen, cleanup, act } from '../../../common/utils/TestUtils.react18';
import { RunViewMetricChartsV2 } from './RunViewMetricChartsV2';
import { DeepPartial } from 'redux';
import { ReduxState } from '../../../redux-types';
import { type RunsChartsBarChartCardProps } from '../runs-charts/components/cards/RunsChartsBarChartCard';
import { RunsChartsLineChartCardProps } from '../runs-charts/components/cards/RunsChartsLineChartCard';
import userEvent from '@testing-library/user-event-14';
import { shouldUseRegexpBasedChartFiltering } from '../../../common/utils/FeatureUtils';
import type { MetricEntitiesByName } from '../../types';

// Mock plot components, as they are not relevant to this test and would hog a lot of resources
jest.mock('../runs-charts/components/cards/RunsChartsBarChartCard', () => ({
  RunsChartsBarChartCard: ({ config }: RunsChartsBarChartCardProps) => (
    <div data-testid="test-bar-plot">Bar plot for {config.metricKey}</div>
  ),
}));
jest.mock('../runs-charts/components/cards/RunsChartsLineChartCard', () => ({
  RunsChartsLineChartCard: ({ config }: RunsChartsLineChartCardProps) => {
    return <div data-testid="test-line-plot">Line plot for {config.metricKey}</div>;
  },
}));

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../common/utils/FeatureUtils'),
  shouldUseRegexpBasedChartFiltering: jest.fn().mockReturnValue(false),
}));

const testRunUuid = 'test_run_uuid';
const testMetricKeys = ['metric_1', 'metric_2', 'system/gpu_1', 'system/gpu_2'];

const testMetrics: MetricEntitiesByName = {
  metric_1: {
    key: 'metric_1',
    step: 0,
    timestamp: 0,
    value: 1000,
  },
  metric_2: {
    key: 'metric_2',
    step: 5,
    timestamp: 0,
    value: 2000,
  },
  'system/gpu_1': {
    key: 'system/gpu_1',
    step: 10,
    timestamp: 10,
    value: 2000,
  },
  'system/gpu_2': {
    key: 'system/gpu_1',
    step: 10,
    timestamp: 10,
    value: 2000,
  },
};

describe('RunViewMetricChartsV2', () => {
  beforeEach(() => {
    localStorage.clear();
  });
  const renderComponent = ({
    mode = 'model',
    metricKeys = testMetricKeys,
    metrics = testMetrics,
  }: {
    mode?: 'model' | 'system';
    metricKeys?: string[];
    metrics?: MetricEntitiesByName;
  } = {}) => {
    const runInfo = {
      runUuid: testRunUuid,
    } as any;
    return render(
      <RunViewMetricChartsV2 runInfo={runInfo} metricKeys={metricKeys} mode={mode} latestMetrics={metrics} />,
      {
        wrapper: ({ children }) => (
          <MockedReduxStoreProvider state={{ entities: { sampledMetricsByRunUuid: {}, imagesByRunUuid: {} } }}>
            <IntlProvider locale="en">{children}</IntlProvider>
          </MockedReduxStoreProvider>
        ),
      },
    );
  };

  it('renders bar charts for two model metrics', async () => {
    renderComponent();
    expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    expect(screen.getByText('Line plot for metric_2')).toBeInTheDocument();

    expect(screen.queryAllByTestId('test-bar-plot')).toHaveLength(1);
    expect(screen.queryAllByTestId('test-line-plot')).toHaveLength(1);
  });

  it('renders line charts for two system metrics', async () => {
    renderComponent({ mode: 'system' });
    expect(screen.getByText('Line plot for system/gpu_1')).toBeInTheDocument();
    expect(screen.getByText('Line plot for system/gpu_2')).toBeInTheDocument();

    expect(screen.queryAllByTestId('test-bar-plot')).toHaveLength(0);
    expect(screen.queryAllByTestId('test-line-plot')).toHaveLength(2);
  });

  it('renders no charts when there are no metric keys configured', async () => {
    renderComponent({
      mode: 'system',
      metricKeys: [],
      metrics: {},
    });

    expect(screen.queryAllByTestId('test-bar-plot')).toHaveLength(0);
    expect(screen.queryAllByTestId('test-line-plot')).toHaveLength(0);

    expect(screen.getByText('No charts in this section')).toBeInTheDocument();
  });

  it('filters metric charts by name (simple)', async () => {
    renderComponent();
    expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    expect(screen.getByText('Line plot for metric_2')).toBeInTheDocument();

    // Filter out one particular chart
    await userEvent.type(screen.getByRole('searchbox'), 'metric_2');
    expect(screen.queryByText('Bar plot for metric_1')).not.toBeInTheDocument();

    // Filter out all charts
    await userEvent.type(screen.getByRole('searchbox'), 'some_metric');
    expect(screen.getByText(/All charts are filtered/)).toBeInTheDocument();
  });

  it('filters metric charts by name (regexp)', async () => {
    jest.mocked(shouldUseRegexpBasedChartFiltering).mockReturnValue(true);
    jest.useFakeTimers();
    const userEventWithTimers = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

    renderComponent();
    expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    expect(screen.getByText('Line plot for metric_2')).toBeInTheDocument();

    // Filter out one particular chart using regexp
    // Note: in RTL, we need to use [[ to represent [
    await userEventWithTimers.type(screen.getByRole('searchbox'), 'm.tric_[[2]$');

    // Wait for filter input debounce
    act(() => {
      jest.advanceTimersByTime(300);
    });

    expect(screen.queryByText('Line plot for metric_2')).toBeInTheDocument();
    expect(screen.queryByText('Bar plot for metric_1')).not.toBeInTheDocument();

    jest.useRealTimers();
  });

  it('adds new charts and sections when new metrics are detected', async () => {
    renderComponent();
    // Assert charts for base metrics only
    expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    expect(screen.getByText('Line plot for metric_2')).toBeInTheDocument();
    expect(screen.queryByText(/plot for metric_3/)).not.toBeInTheDocument();
    expect(screen.queryByText(/plot for metric_4/)).not.toBeInTheDocument();

    // Ummount the component
    cleanup();

    const newMetrics = {
      ...testMetrics,
      metric_3: {
        key: 'metric_3',
        step: 5,
        timestamp: 0,
        value: 1000,
      },
      metric_4: {
        key: 'metric_4',
        step: 0,
        timestamp: 0,
        value: 1000,
      },
      'custom-prefix/test-metric': {
        key: 'custom-prefix/test-metric',
        step: 0,
        timestamp: 0,
        value: 1000,
      },
    };

    renderComponent({
      metrics: newMetrics,
    });

    // Assert new charts
    expect(screen.getByText('Line plot for metric_3')).toBeInTheDocument();
    expect(screen.getByText('Bar plot for metric_4')).toBeInTheDocument();
    expect(screen.getByText('Bar plot for custom-prefix/test-metric')).toBeInTheDocument();

    // Assert new section
    expect(screen.getByText('custom-prefix')).toBeInTheDocument();
  });
});

import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { render, screen, cleanup, act, waitFor } from '../../../common/utils/TestUtils.react18';
import { RunViewMetricCharts } from './RunViewMetricCharts';
import { DeepPartial } from 'redux';
import { ReduxState } from '../../../redux-types';
import { type RunsChartsBarChartCardProps } from '../runs-charts/components/cards/RunsChartsBarChartCard';
import type { RunsChartsLineChartCardProps } from '../runs-charts/components/cards/RunsChartsLineChartCard';
import userEvent from '@testing-library/user-event';
import type { MetricEntitiesByName } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import { DesignSystemProvider } from '@databricks/design-system';
import { MlflowService } from '../../sdk/MlflowService';
import { LOG_IMAGE_TAG_INDICATOR } from '../../constants';

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

// Mock useIsInViewport hook to simulate that the chart element is in the viewport
jest.mock('../runs-charts/hooks/useIsInViewport', () => ({
  useIsInViewport: () => ({ isInViewport: true, setElementRef: jest.fn() }),
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

describe('RunViewMetricCharts', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.spyOn(MlflowService, 'listArtifacts').mockImplementation(() => Promise.resolve([]));
  });
  const renderComponent = ({
    mode = 'model',
    metricKeys = testMetricKeys,
    metrics = testMetrics,
    tags = {},
  }: {
    mode?: 'model' | 'system';
    metricKeys?: string[];
    metrics?: MetricEntitiesByName;
    tags?: Record<string, KeyValueEntity>;
  } = {}) => {
    const runInfo = {
      runUuid: testRunUuid,
    } as any;
    return render(
      <RunViewMetricCharts runInfo={runInfo} metricKeys={metricKeys} mode={mode} latestMetrics={metrics} tags={tags} />,
      {
        wrapper: ({ children }) => (
          <DesignSystemProvider>
            <MockedReduxStoreProvider state={{ entities: { sampledMetricsByRunUuid: {}, imagesByRunUuid: {} } }}>
              <IntlProvider locale="en">{children}</IntlProvider>
            </MockedReduxStoreProvider>
          </DesignSystemProvider>
        ),
      },
    );
  };

  it('renders bar charts for two model metrics', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    });
    expect(screen.getByText('Line plot for metric_2')).toBeInTheDocument();

    expect(screen.queryAllByTestId('test-bar-plot')).toHaveLength(1);
    expect(screen.queryAllByTestId('test-line-plot')).toHaveLength(1);
  });

  it('renders line charts for two system metrics', async () => {
    renderComponent({ mode: 'system' });

    await waitFor(() => {
      expect(screen.getByText('Line plot for system/gpu_1')).toBeInTheDocument();
    });

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

    await waitFor(() => {
      expect(screen.queryAllByTestId('test-bar-plot')).toHaveLength(0);
      expect(screen.queryAllByTestId('test-line-plot')).toHaveLength(0);

      expect(screen.getByText('No charts in this section')).toBeInTheDocument();
    });
  });

  it('filters metric charts by name (regexp)', async () => {
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
    await waitFor(() => {
      expect(screen.getByText('Bar plot for metric_1')).toBeInTheDocument();
    });

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
    await waitFor(() => {
      expect(screen.getByText('Line plot for metric_3')).toBeInTheDocument();
    });
    expect(screen.getByText('Bar plot for metric_4')).toBeInTheDocument();
    expect(screen.getByText('Bar plot for custom-prefix/test-metric')).toBeInTheDocument();

    // Assert new section
    expect(screen.getByText('custom-prefix')).toBeInTheDocument();
  });

  it('adds should not call for image artifacts when `mlflow.loggedImages` tag is not set', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Model metrics')).toBeInTheDocument();
    });

    expect(MlflowService.listArtifacts).not.toHaveBeenCalled();
  });

  it('adds should call for image artifacts when `mlflow.loggedImages` tag is set', async () => {
    renderComponent({
      tags: {
        [LOG_IMAGE_TAG_INDICATOR]: {
          key: LOG_IMAGE_TAG_INDICATOR,
          value: 'true',
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Model metrics')).toBeInTheDocument();
    });

    expect(MlflowService.listArtifacts).toHaveBeenCalledWith({ path: 'images', run_uuid: 'test_run_uuid' });
  });
});

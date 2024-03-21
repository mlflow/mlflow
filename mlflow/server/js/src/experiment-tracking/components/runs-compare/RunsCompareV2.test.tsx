import { shouldEnableDeepLearningUI, shouldEnableRunGrouping } from '../../../common/utils/FeatureUtils';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { renderWithIntl, act, fireEvent, screen, within } from 'common/utils/TestUtils.react18';
import { MetricEntitiesByName } from '../../types';
import { useUpdateExperimentViewUIState } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import { ExperimentPageUIStateV2 } from '../experiment-page/models/ExperimentPageUIStateV2';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { useMultipleChartsMetricHistory } from './hooks/useMultipleChartsMetricHistory';
import {
  RunsChartType,
  RunsChartsBarCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
} from '../runs-charts/runs-charts.types';
import { RunsCompareV2 } from './RunsCompareV2';
import { useSampledMetricHistory } from '../runs-charts/hooks/useSampledMetricHistory';
import userEvent from '@testing-library/user-event-14';
import { RunsChartsLineChartXAxisType } from '../runs-charts/components/RunsCharts.common';

jest.setTimeout(30000); // Larger timeout for integration testing

// Force-enable drag and drop for this test
jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(),
  shouldEnableRunGrouping: jest.fn(),
}));

// Mock the chart component to save time on rendering
jest.mock('../runs-charts/components/RunsMetricsBarPlot', () => ({
  RunsMetricsBarPlot: () => <div />,
}));
jest.mock('../runs-charts/components/RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: () => <div />,
}));

// Mock the UI State handler
jest.mock('../experiment-page/contexts/ExperimentPageUIStateContext', () => ({
  useUpdateExperimentViewUIState: jest.fn(),
}));

jest.mock('./hooks/useMultipleChartsMetricHistory', () => ({
  useMultipleChartsMetricHistory: jest.fn(),
}));

jest.mock('../runs-charts/hooks/useSampledMetricHistory', () => ({
  useSampledMetricHistory: jest.fn(),
}));

jest.setTimeout(30000); // Larger timeout for integration testing

describe('RunsCompareV2', () => {
  const testCharts: (RunsChartsParallelCardConfig | RunsChartsBarCardConfig)[] = [
    {
      type: RunsChartType.PARALLEL,
      uuid: 'chart-parallel',
      runsCountToCompare: 10,
      selectedMetrics: [],
      selectedParams: [],
      metricSectionId: 'metric-section-1',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-alpha',
      runsCountToCompare: 10,
      metricKey: 'metric-alpha',
      metricSectionId: 'metric-section-1',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-beta',
      runsCountToCompare: 10,
      metricKey: 'metric-beta',
      metricSectionId: 'metric-section-1',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-gamma',
      runsCountToCompare: 10,
      metricKey: 'metric-gamma',
      metricSectionId: 'metric-section-1',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-omega',
      runsCountToCompare: 10,
      metricKey: 'tmp/metric-omega',
      metricSectionId: 'metric-section-0',
      deleted: false,
      isGenerated: true,
    },
  ];

  const testMultipleMetricsLineChart: RunsChartsLineCardConfig = {
    type: RunsChartType.LINE,
    runsCountToCompare: 10,
    metricSectionId: 'metric-section-1',
    deleted: false,
    isGenerated: true,
    uuid: 'two-metric-line-chart',
    metricKey: '',
    lineSmoothness: 0,
    scaleType: 'linear',
    xAxisKey: RunsChartsLineChartXAxisType.STEP,
    selectedXAxisMetricKey: '',
    selectedMetricKeys: ['metric-beta', 'metric-alpha'],
  };

  const compareRunSections = [
    {
      uuid: 'metric-section-0',
      name: 'tmp',
      display: true,
      isReordered: false,
    },
    {
      uuid: 'metric-section-1',
      name: 'Model metrics',
      display: true,
      isReordered: false,
    },
    {
      uuid: 'metric-section-2',
      name: 'System metrics',
      display: true,
      isReordered: false,
    },
  ];

  let currentUIState = {} as ExperimentPageUIStateV2;

  const updateUIState = jest.fn().mockImplementation((uiStateTransformer) => {
    currentUIState = uiStateTransformer(currentUIState);
  });

  beforeEach(() => {
    jest.mocked(shouldEnableDeepLearningUI).mockReturnValue(true);
    jest.mocked(shouldEnableRunGrouping).mockReturnValue(false);
    jest.mocked(useUpdateExperimentViewUIState).mockReturnValue(updateUIState);
    jest.mocked(useMultipleChartsMetricHistory).mockReturnValue({
      isLoading: false,
      chartRunDataWithHistory: [],
    });
    jest.mocked(useSampledMetricHistory).mockReturnValue({
      isLoading: false,
      isRefreshing: false,
      resultsByRunUuid: {},
      refresh: jest.fn(),
    });
    currentUIState.compareRunCharts = testCharts;
    currentUIState.compareRunSections = compareRunSections;
    currentUIState.isAccordionReordered = false;
    updateUIState.mockClear();
  });

  const createComponentMock = ({
    comparedRuns = [],
    latestMetricsByRunUuid = {},
    groupBy = '',
  }: {
    comparedRuns?: RunRowType[];
    latestMetricsByRunUuid?: Record<string, MetricEntitiesByName>;
    groupBy?: string;
  } = {}) => {
    return renderWithIntl(
      <Provider
        store={configureStore([thunk, promiseMiddleware()])({
          entities: { paramsByRunUuid: {}, latestMetricsByRunUuid },
        })}
      >
        <RunsCompareV2
          comparedRuns={comparedRuns}
          experimentTags={{}}
          isLoading={false}
          metricKeyList={[]}
          paramKeyList={[]}
          compareRunCharts={currentUIState.compareRunCharts}
          compareRunSections={currentUIState.compareRunSections}
          groupBy={groupBy}
        />
      </Provider>,
    );
  };

  const getChartArea = (chartTitle: string) => {
    const firstMetricHeading = screen.getByRole('heading', { name: chartTitle });
    return firstMetricHeading.closest('[data-testid="experiment-view-compare-runs-card"]') as HTMLElement;
  };

  const getSectionArea = (sectionName: string) => {
    const firstSectionHeading = screen.getByText(sectionName);
    return firstSectionHeading.closest('[data-testid="experiment-view-compare-runs-section-header"]') as HTMLElement;
  };

  test('should render multiple charts and reorder using drag and drop', async () => {
    await act(async () => {
      createComponentMock();
    });

    const betaChartArea = getChartArea('metric-beta');
    const alphaChartArea = getChartArea('metric-alpha');
    const gammaChartArea = getChartArea('metric-gamma');

    const betaHandle = within(betaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');

    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-alpha',
      'chart-beta',
      'chart-gamma',
      'chart-omega',
    ]);

    // Drag "beta" chart into the "alpha" chart position
    await act(async () => {
      fireEvent.dragStart(betaHandle);
      fireEvent.dragEnter(alphaChartArea);
      fireEvent.drop(alphaChartArea);
    });

    // Verify that the charts are reordered
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-beta',
      'chart-alpha',
      'chart-gamma',
      'chart-omega',
    ]);

    const gammaHandle = within(gammaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag "gamma" chart into the "beta" chart position
    await act(async () => {
      fireEvent.dragStart(gammaHandle);
      fireEvent.dragEnter(betaChartArea);
      fireEvent.drop(betaChartArea);
    });

    // Verify that the charts are reordered
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-gamma',
      'chart-beta',
      'chart-alpha',
      'chart-omega',
    ]);

    // Drag "gamma" chart into the "alpha" chart position
    await act(async () => {
      fireEvent.dragStart(gammaHandle);
      fireEvent.dragEnter(alphaChartArea);
      fireEvent.drop(alphaChartArea);
    });

    // Verify that the charts are reordered
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-beta',
      'chart-alpha',
      'chart-gamma',
      'chart-omega',
    ]);
  });

  test('drag and drop (reorder) across sections', async () => {
    await act(async () => {
      createComponentMock();
    });

    const betaChartArea = getChartArea('metric-beta');
    const alphaChartArea = getChartArea('metric-alpha');
    const omegaChartArea = getChartArea('tmp/metric-omega');

    const betaHandle = within(betaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');
    // Drag "beta" chart into the "omega" chart position
    await act(async () => {
      fireEvent.dragStart(betaHandle);
      fireEvent.dragEnter(omegaChartArea);
      fireEvent.drop(omegaChartArea);
    });

    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-alpha',
      'chart-gamma',
      'chart-beta',
      'chart-omega',
    ]);

    const omegaHandle = within(omegaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');
    // Drag "omega" chart into the "alpha" chart position
    await act(async () => {
      fireEvent.dragStart(omegaHandle);
      fireEvent.dragEnter(alphaChartArea);
      fireEvent.drop(alphaChartArea);
    });

    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-omega',
      'chart-alpha',
      'chart-gamma',
      'chart-beta',
    ]);

    expect(currentUIState.compareRunCharts?.map(({ metricSectionId }) => metricSectionId)).toEqual([
      'metric-section-1',
      'metric-section-1',
      'metric-section-1',
      'metric-section-1',
      'metric-section-0',
    ]);
  });

  test('initializes correct chart types for given initial runs data', async () => {
    currentUIState.compareRunCharts = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run', runInfo: {} },
          { runUuid: 'run_oldest', runName: 'First run', runInfo: {} },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'metric-with-history': { key: 'metric-with-history', value: 1, step: 1 },
            'metric-static': { key: 'metric-static', value: 1, step: 0 },
          },
          run_oldest: {
            'metric-with-history': { key: 'metric-with-history', value: 1, step: 5 },
            'metric-static': { key: 'metric-static', value: 1, step: 0 },
          },
        } as any,
      });
    });

    expect(updateUIState).toHaveBeenCalled();

    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'metric-static',
        type: 'BAR',
      }),
      // "metric-with-history" should be initialized as a line chart since there's at least one run with the history
      expect.objectContaining({
        metricKey: 'metric-with-history',
        type: 'LINE',
      }),
    ]);
  });

  test('initializes sections', async () => {
    currentUIState.compareRunCharts = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run' },
          { runUuid: 'run_oldest', runName: 'First run' },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 1 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
          },
          run_oldest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
          },
        } as any,
      });
    });
    expect(updateUIState).toHaveBeenCalled();

    expect(currentUIState.compareRunSections).toEqual([
      expect.objectContaining({
        name: 'tmp',
        display: true,
      }),
      expect.objectContaining({
        name: 'Model metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'System metrics',
        display: true,
      }),
    ]);
  });

  test('drag and drop sections', async () => {
    await act(async () => {
      createComponentMock();
    });

    expect(currentUIState.compareRunSections?.map(({ uuid }) => uuid)).toEqual([
      'metric-section-0',
      'metric-section-1',
      'metric-section-2',
    ]);

    const metricSection0 = getSectionArea('tmp');
    const metricSection1 = getSectionArea('Model metrics');
    const metricSection2 = getSectionArea('System metrics');
    const metricSection0Handle = within(metricSection0).getByTestId(
      'experiment-view-compare-runs-section-header-drag-handle',
    );

    // Move section 'tmp' to section 'System metrics'
    await act(async () => {
      fireEvent.dragStart(metricSection0Handle);
      fireEvent.dragEnter(metricSection2);
      fireEvent.dragOver(metricSection2);
      fireEvent.drop(metricSection2);
    });

    expect(currentUIState.compareRunSections?.map(({ uuid }) => uuid)).toEqual([
      'metric-section-1',
      'metric-section-2',
      'metric-section-0',
    ]);

    const metricSection2Handle = within(metricSection2).getByTestId(
      'experiment-view-compare-runs-section-header-drag-handle',
    );

    // Move section 'System metrics' to section 'Model metrics'
    await act(async () => {
      fireEvent.dragStart(metricSection2Handle);
      fireEvent.dragEnter(metricSection1);
      fireEvent.dragOver(metricSection1);
      fireEvent.drop(metricSection1);
    });

    expect(currentUIState.compareRunSections?.map(({ uuid }) => uuid)).toEqual([
      'metric-section-2',
      'metric-section-1',
      'metric-section-0',
    ]);
  });

  test('detecting and adding new sections when already in default order', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run', runInfo: { run_uuid: 'run_latest' } },
          { runUuid: 'run_oldest', runName: 'First run', runInfo: { run_uuid: 'run_oldest' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 1 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_oldest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunSections).toEqual([
      expect.objectContaining({
        name: 'tmp',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp2',
        display: true,
      }),
      expect.objectContaining({
        name: 'Model metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'System metrics',
        display: true,
      }),
    ]);

    expect(currentUIState.isAccordionReordered).toBe(false);

    // New run is added to the end of the list
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run', runInfo: { run_uuid: 'run_latest' } },
          { runUuid: 'run_oldest', runName: 'First run', runInfo: { run_uuid: 'run_oldest' } },
          { runUuid: 'run_new', runName: 'New run', runInfo: { run_uuid: 'run_new' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 1 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_oldest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_new: {
            'tmp1/metric-static': { key: 'tmp1/metric-static', value: 1, step: 5 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunSections).toEqual([
      expect.objectContaining({
        name: 'tmp',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp1',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp2',
        display: true,
      }),
      expect.objectContaining({
        name: 'Model metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'System metrics',
        display: true,
      }),
    ]);
  });

  test('detecting and adding new sections when not in default order', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run', runInfo: { run_uuid: 'run_latest' } },
          { runUuid: 'run_oldest', runName: 'First run', runInfo: { run_uuid: 'run_oldest' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 1 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_oldest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunSections).toEqual([
      expect.objectContaining({
        name: 'tmp',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp2',
        display: true,
      }),
      expect.objectContaining({
        name: 'Model metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'System metrics',
        display: true,
      }),
    ]);

    // Rerender with compareRunCharts and compareRunSections initialized
    await act(async () => {
      createComponentMock();
    });

    // Update order of compareRunSections
    const tmpSection = getSectionArea('tmp');
    const tmp2Section = getSectionArea('tmp2');
    const tmpSectionHandle = within(tmpSection).getByTestId('experiment-view-compare-runs-section-header-drag-handle');
    // Move section 'tmp' to section 'tmp2'
    await act(async () => {
      fireEvent.dragStart(tmpSectionHandle);
      fireEvent.dragEnter(tmp2Section);
      fireEvent.dragOver(tmp2Section);
      fireEvent.drop(tmp2Section);
    });

    expect(currentUIState.isAccordionReordered).toBe(true);

    // New run is added to the end of the list
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runName: 'Last run', runInfo: { run_uuid: 'run_latest' } },
          { runUuid: 'run_oldest', runName: 'First run', runInfo: { run_uuid: 'run_oldest' } },
          { runUuid: 'run_new', runName: 'New run', runInfo: { run_uuid: 'run_new' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 1 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_oldest: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
            'tmp/metric-static': { key: 'tmp/metric-static', value: 1, step: 1 },
            'tmp2/metric-static': { key: 'tmp2/metric-static', value: 1, step: 1 },
          },
          run_new: {
            'tmp1/metric-static': { key: 'tmp1/metric-static', value: 1, step: 5 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunSections).toEqual([
      expect.objectContaining({
        name: 'tmp2',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp',
        display: true,
      }),
      expect.objectContaining({
        name: 'Model metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'System metrics',
        display: true,
      }),
      expect.objectContaining({
        name: 'tmp1',
        display: true,
      }),
    ]);
  });

  test('detecting metric history and updating bar chart to line chart', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    // Run with static metric
    await act(async () => {
      createComponentMock({
        comparedRuns: [{ runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } }] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 0 },
          },
        } as any,
      });
    });

    // Test that the chart is initialized as a bar chart
    expect((currentUIState.compareRunCharts || []).map(({ type }) => type)).toEqual(['BAR']);

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 0 },
          },
          run_2: {
            'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 5 },
          },
        } as any,
      });
    });

    // Test that the chart is updated to a line chart
    expect((currentUIState.compareRunCharts || []).map(({ type }) => type)).toEqual(['LINE']);
  });

  test('detecting new run and inserting new chart when section is not reordered', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section1/metric6': { key: 'section1/metric6', value: 1 },
          },
        } as any,
      });
    });

    // Should be sorted when initialized
    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric6',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
    ]);

    // Inserting a new run with section1/metric4
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
          { runUuid: 'run_3', runName: 'Third run', runInfo: { run_uuid: 'run_3' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section1/metric6': { key: 'section1/metric6', value: 1 },
          },
          run_3: {
            'section1/metric4': { key: 'section1/metric4', value: 1 },
          },
        } as any,
      });
    });

    // Should be sorted when initialized
    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric4',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric6',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
    ]);
  });

  test('detecting new run and inserting new chart when section is reordered by dnd within', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section1/metric6', value: 1 },
          },
        } as any,
      });
    });
    // Rerender with compareRunCharts and compareRunSections initialized
    await act(async () => {
      createComponentMock();
    });

    // Should be sorted when initialized
    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
      expect.objectContaining({
        metricKey: 'section2/metric6',
      }),
    ]);

    const chartArea3 = getChartArea('section1/metric3');
    const chartArea5 = getChartArea('section1/metric5');

    // Get section1/metric3 handle
    const chartArea3Handle = within(chartArea3).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag "section1/metric3" chart into the "section1/metric5" chart position
    await act(async () => {
      fireEvent.dragStart(chartArea3Handle);
      fireEvent.dragEnter(chartArea5);
      fireEvent.drop(chartArea5);
    });

    // Adding a new run with section1/metric6 should append the metric at the end
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
          { runUuid: 'run_3', runName: 'Third run', runInfo: { run_uuid: 'run_3' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section1/metric6', value: 1 },
          },
          run_3: {
            'section1/metric6': { key: 'section1/metric6', value: 1 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
      expect.objectContaining({
        metricKey: 'section2/metric6',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric6',
      }),
    ]);
  });

  test('detecting new run and inserting new chart when section is reordered by dragging chart out', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section2/metric6', value: 1 },
          },
        } as any,
      });
    });
    // Rerender with compareRunCharts and compareRunSections initialized
    await act(async () => {
      createComponentMock();
    });

    // Drag section1/metric5 into section2
    const chartArea5 = getChartArea('section1/metric5');
    const chartArea6 = getChartArea('section2/metric6');

    // Get section1/metric5 handle
    const chartArea5Handle = within(chartArea5).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag section1/metric5 chart into section2/metric6 chart position
    await act(async () => {
      fireEvent.dragStart(chartArea5Handle);
      fireEvent.dragEnter(chartArea6);
      fireEvent.drop(chartArea6);
    });

    // Adding a new run with section1/metric4 should append the metric at the end
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
          { runUuid: 'run_3', runName: 'Third run', runInfo: { run_uuid: 'run_3' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section2/metric6', value: 1 },
          },
          run_3: {
            'section1/metric4': { key: 'section1/metric4', value: 1 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section2/metric6',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric4',
      }),
    ]);
  });

  test('detecting new run and inserting new chart when section is reordered by dragging chart in', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section2/metric6', value: 1 },
          },
        } as any,
      });
    });
    // Rerender with compareRunCharts and compareRunSections initialized
    await act(async () => {
      createComponentMock();
    });

    // Drag section2/metric6 into section1
    const chartArea6 = getChartArea('section2/metric6');
    const chartArea5 = getChartArea('section1/metric5');

    // Get section2/metric6 handle
    const chartArea6Handle = within(chartArea6).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag "section2/metric6" chart into the "section1/metric5" chart position
    await act(async () => {
      fireEvent.dragStart(chartArea6Handle);
      fireEvent.dragEnter(chartArea5);
      fireEvent.drop(chartArea5);
    });

    // Adding a new run with section1/metric4 should append the metric at the end
    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_1', runName: 'First run', runInfo: { run_uuid: 'run_1' } },
          { runUuid: 'run_2', runName: 'Second run', runInfo: { run_uuid: 'run_2' } },
          { runUuid: 'run_3', runName: 'Third run', runInfo: { run_uuid: 'run_3' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            'section1/metric3': { key: 'section1/metric3', value: 1 },
            'section1/metric7': { key: 'section1/metric7', value: 1 },
            'section1/metric5': { key: 'section1/metric5', value: 1 },
          },
          run_2: {
            'section2/metric6': { key: 'section2/metric6', value: 1 },
          },
          run_3: {
            'section1/metric4': { key: 'section1/metric4', value: 1 },
          },
        } as any,
      });
    });

    expect(currentUIState.compareRunCharts).toEqual([
      expect.objectContaining({
        metricKey: 'section1/metric3',
      }),
      expect.objectContaining({
        metricKey: 'section2/metric6',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric5',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric7',
      }),
      expect.objectContaining({
        metricKey: 'section1/metric4',
      }),
    ]);
  });

  test('search functionality filters metric charts', async () => {
    // For this test, add chart with multiple metrics at the end
    currentUIState.compareRunCharts = [...testCharts, testMultipleMetricsLineChart];

    await act(async () => {
      createComponentMock();
    });
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).toBeInTheDocument();

    // Paste search query into searchbox
    await userEvent.click(screen.getByRole('searchbox'));
    await userEvent.paste('alpha');

    const modelMetricsSection = screen.getByText('Model metrics');
    const sectionHeader = modelMetricsSection.closest(
      '[data-testid="on-search-runs-compare-section-header"]',
    ) as HTMLElement;

    // Click on the section header
    await act(async () => {
      fireEvent.click(sectionHeader);
    });

    expect(screen.queryByRole('heading', { name: 'metric-beta' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).toBeInTheDocument();
  });

  test('correctly determines chart types run groups mixed with ungrouped runs', async () => {
    jest.mocked(shouldEnableRunGrouping).mockReturnValue(true);

    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    // Render a component with a group and a run:
    // - a group contains aggregated "metric_1" data indicating history
    // - a group contains aggregated "metric_2" data not indicating any history
    // - a run contains "metric_3" data indicating history
    await act(async () => {
      createComponentMock({
        groupBy: 'metric_1',
        comparedRuns: [
          {
            runUuid: '',
            groupParentInfo: {
              groupId: 'some_group',
              aggregatedMetricData: {
                metric_1: {
                  key: 'metric_1',
                  value: 123,
                  maxStep: 5,
                },
                metric_2: {
                  key: 'metric_2',
                  value: 123,
                  maxStep: 0,
                },
              },
            },
          },
          {
            runUuid: 'run_1',
            runName: 'First run',
            runInfo: { run_uuid: 'run_1' },
            runDateAndNestInfo: { belongsToGroup: false },
          },
        ] as any,
        latestMetricsByRunUuid: {
          run_1: {
            metric_3: { key: 'metric_3', value: 1, step: 3 },
          },
        } as any,
      });
    });

    // Assert correct chart types for metrics
    expect(currentUIState.compareRunCharts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          metricKey: 'metric_1',
          type: 'LINE',
        }),
        expect.objectContaining({
          metricKey: 'metric_2',
          type: 'BAR',
        }),
        expect.objectContaining({
          metricKey: 'metric_3',
          type: 'LINE',
        }),
      ]),
    );
  });
});

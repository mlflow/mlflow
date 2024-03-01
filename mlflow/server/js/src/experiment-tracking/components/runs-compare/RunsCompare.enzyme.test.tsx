import { shouldEnableDeepLearningUI } from '../../../common/utils/FeatureUtils';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, act, fireEvent, screen, within } from 'common/utils/TestUtils.react17';
import { MetricEntitiesByName } from '../../types';
// import { SearchExperimentRunsFacetsState } from '../experiment-page/models/SearchExperimentRunsFacetsState';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsCompare } from './RunsCompare';
import { RunsChartType, RunsChartsBarCardConfig, RunsChartsParallelCardConfig } from '../runs-charts/runs-charts.types';
import { createExperimentPageUIState } from '../experiment-page/models/ExperimentPageUIState';
import { ExperimentPageUIStateContextProvider } from '../experiment-page/contexts/ExperimentPageUIStateContext';

// Force-enable drag and drop for this test
jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(),
}));

// Mock the chart component to save time on rendering
jest.mock('../runs-charts/components/RunsMetricsBarPlot', () => ({
  RunsMetricsBarPlot: () => <div />,
}));

describe('RunsCompare', () => {
  const testCharts: (RunsChartsParallelCardConfig | RunsChartsBarCardConfig)[] = [
    {
      type: RunsChartType.PARALLEL,
      uuid: 'chart-parallel',
      runsCountToCompare: 10,
      selectedMetrics: [],
      selectedParams: [],
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-alpha',
      runsCountToCompare: 10,
      metricKey: 'metric-alpha',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-beta',
      runsCountToCompare: 10,
      metricKey: 'metric-beta',
      deleted: false,
      isGenerated: true,
    },
    {
      type: RunsChartType.BAR,
      uuid: 'chart-gamma',
      runsCountToCompare: 10,
      metricKey: 'metric-gamma',
      deleted: false,
      isGenerated: true,
    },
  ];
  let currentUIState = createExperimentPageUIState();

  const updateUIState = jest.fn().mockImplementation((facetsTransformer) => {
    currentUIState = facetsTransformer(currentUIState);
  });

  beforeEach(() => {
    jest.mocked(shouldEnableDeepLearningUI).mockReturnValue(true);
    currentUIState.compareRunCharts = testCharts;
    updateUIState.mockClear();
  });

  const createComponentMock = ({
    comparedRuns = [],
    latestMetricsByRunUuid = {},
  }: { comparedRuns?: RunRowType[]; latestMetricsByRunUuid?: Record<string, MetricEntitiesByName> } = {}) => {
    return renderWithIntl(
      <MockedReduxStoreProvider state={{ entities: { paramsByRunUuid: {}, latestMetricsByRunUuid } }}>
        <ExperimentPageUIStateContextProvider setUIState={updateUIState}>
          <RunsCompare
            comparedRuns={comparedRuns}
            experimentTags={{}}
            isLoading={false}
            metricKeyList={[]}
            paramKeyList={[]}
            compareRunCharts={currentUIState.compareRunCharts}
          />
        </ExperimentPageUIStateContextProvider>
      </MockedReduxStoreProvider>,
    );
  };

  const getChartArea = (chartTitle: string) => {
    const firstMetricHeading = screen.getByRole('heading', { name: chartTitle });
    return firstMetricHeading.closest('[data-testid="experiment-view-compare-runs-card"]') as HTMLElement;
  };

  test('should render multiple charts and reorder them using drag and drop', async () => {
    await act(async () => {
      createComponentMock();
    });

    expect(updateUIState).not.toHaveBeenCalled();

    const betaChartArea = getChartArea('metric-beta');
    const alphaChartArea = getChartArea('metric-alpha');
    const gammaChartArea = getChartArea('metric-gamma');

    const betaHandle = within(betaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');

    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-alpha',
      'chart-beta',
      'chart-gamma',
    ]);

    // Drag "beta" chart into the "alpha" chart position
    await act(async () => {
      fireEvent.dragStart(betaHandle);
      fireEvent.dragEnter(alphaChartArea);
      fireEvent.drop(alphaChartArea);
    });

    // Verify that the charts are reordered, and the "beta" chart is now in the "alpha" chart position
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-beta',
      'chart-alpha',
      'chart-gamma',
    ]);

    const gammaHandle = within(gammaChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag "gamma" chart into the "beta" chart position
    await act(async () => {
      fireEvent.dragStart(gammaHandle);
      fireEvent.dragEnter(betaChartArea);
      fireEvent.drop(betaChartArea);
    });

    // Verify that the charts are reordered, and the "gamma" chart is now in the "beta" chart position
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-gamma',
      'chart-alpha',
      'chart-beta',
    ]);
  });

  test('should prevent from mixing up charts from non-compatible areas', async () => {
    await act(async () => {
      createComponentMock();
    });

    expect(updateUIState).not.toHaveBeenCalled();

    const parallelChartArea = getChartArea('Parallel Coordinates');
    const parallelChartHandle = within(parallelChartArea).getByTestId('experiment-view-compare-runs-card-drag-handle');
    const betaChartArea = getChartArea('metric-beta');

    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-alpha',
      'chart-beta',
      'chart-gamma',
    ]);

    // Drag "beta" chart into the "alpha" chart position
    await act(async () => {
      fireEvent.dragStart(parallelChartHandle);
      fireEvent.dragEnter(betaChartArea);
      fireEvent.drop(betaChartArea);
    });

    // Confirm that the charts has not been reordered
    expect(updateUIState).not.toHaveBeenCalled();
    expect(currentUIState.compareRunCharts?.map(({ uuid }) => uuid)).toEqual([
      'chart-parallel',
      'chart-alpha',
      'chart-beta',
      'chart-gamma',
    ]);
  });

  test('initializes correct chart types for given initial runs data', async () => {
    currentUIState.compareRunCharts = undefined;

    await act(async () => {
      createComponentMock({
        comparedRuns: [
          { runUuid: 'run_latest', runInfo: { run_name: 'Last run' } },
          { runUuid: 'run_oldest', runInfo: { run_name: 'First run' } },
        ] as any,
        latestMetricsByRunUuid: {
          run_latest: {
            'metric-with-history': { key: 'metric-with-history', value: 1, step: 0 },
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
      // "metric-with-history" should be initialized as a line chart since there's at least one run with the history
      expect.objectContaining({
        metricKey: 'metric-with-history',
        type: 'LINE',
      }),
      expect.objectContaining({
        metricKey: 'metric-static',
        type: 'BAR',
      }),
    ]);
  });
});

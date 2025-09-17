import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import {
  renderWithIntl,
  act,
  fireEvent,
  screen,
  within,
  waitFor,
  cleanup,
} from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ImageEntity, MetricEntitiesByName } from '../../types';
import { ExperimentPageUIStateContextProvider } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import type {
  ExperimentPageUIState,
  ExperimentRunsChartsUIConfiguration,
} from '../experiment-page/models/ExperimentPageUIState';
import { createExperimentPageUIState } from '../experiment-page/models/ExperimentPageUIState';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import type {
  RunsChartsBarCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsDifferenceCardConfig,
} from '../runs-charts/runs-charts.types';
import { RunsChartType, DifferenceCardConfigCompareGroup } from '../runs-charts/runs-charts.types';
import { RunsCompare } from './RunsCompare';
import { useSampledMetricHistory } from '../runs-charts/hooks/useSampledMetricHistory';
import userEvent from '@testing-library/user-event';
import { RunsChartsLineChartXAxisType } from '../runs-charts/components/RunsCharts.common';
import { useState } from 'react';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing

// Mock the chart component to save time on rendering
jest.mock('../runs-charts/components/RunsMetricsBarPlot', () => ({
  RunsMetricsBarPlot: ({ metricKey }: any) => <div>[bar plot for {metricKey}]</div>,
}));
jest.mock('../runs-charts/components/RunsMetricsLinePlot', () => ({
  RunsMetricsLinePlot: ({ metricKey, selectedMetricKeys = [] }: any) => (
    <div>[line plot for {[...selectedMetricKeys, metricKey].filter(Boolean).join(',')}]</div>
  ),
}));
jest.mock('../runs-charts/hooks/useSampledMetricHistory', () => ({
  useSampledMetricHistory: jest.fn(),
}));

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableHidingChartsWithNoData: jest.fn(() => false),
  shouldEnableImageGridCharts: jest.fn(() => true),
}));

// Mock useIsInViewport hook to simulate that the chart element is in the viewport
jest.mock('../runs-charts/hooks/useIsInViewport', () => ({
  useIsInViewport: () => ({ isInViewport: true, setElementRef: jest.fn() }),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing

// Helper function to assert order of HTMLElements
function assertElementsInOrder(elements: HTMLElement[]) {
  for (let i = 0; i < elements.length - 1; i++) {
    const current = elements[i];
    const next = elements[i + 1];

    // Use compareDocumentPosition to check the order
    const position = current.compareDocumentPosition(next);

    // Assert that `next` follows `current`
    expect(position).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  }
}

const testCases = [
  {
    description: '',
    setup: () => {},
  },
];
describe.each(testCases)('RunsCompare $description', ({ setup: testCaseSetup }) => {
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
    xAxisScaleType: 'linear',
    selectedXAxisMetricKey: '',
    selectedMetricKeys: ['metric-beta', 'metric-alpha'],
    range: {
      xMin: undefined,
      xMax: undefined,
      yMin: undefined,
      yMax: undefined,
    },
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

  let currentUIState = {} as ExperimentPageUIState;

  beforeEach(() => {
    testCaseSetup();
    jest.mocked(useSampledMetricHistory).mockReturnValue({
      isLoading: false,
      isRefreshing: false,
      resultsByRunUuid: {},
      refresh: jest.fn(),
    });
    currentUIState = createExperimentPageUIState();
    currentUIState.compareRunCharts = testCharts;
    currentUIState.hideEmptyCharts = false;
    currentUIState.compareRunSections = compareRunSections;
    currentUIState.isAccordionReordered = false;
  });

  const getCurrentUIState = async () => {
    return currentUIState;
  };

  const createComponentMock = ({
    comparedRuns = [],
    latestMetricsByRunUuid = {},
    imagesByRunUuid = {},
    groupBy = '',
  }: {
    comparedRuns?: RunRowType[];
    latestMetricsByRunUuid?: Record<string, MetricEntitiesByName>;
    imagesByRunUuid?: Record<string, Record<string, Record<string, ImageEntity>>>;
    groupBy?: string;
  } = {}) => {
    const TestComponent = () => {
      // Create a contextual uiState based on mutable "currentUIState" variable
      const [uiState, setUIState] = useState<ExperimentPageUIState>(() => currentUIState);

      // With every render, keep the currentUIState updated so it's easy to assert on
      currentUIState = uiState;

      return (
        <ExperimentPageUIStateContextProvider setUIState={setUIState}>
          <RunsCompare
            comparedRuns={comparedRuns}
            experimentTags={{}}
            isLoading={false}
            metricKeyList={[]}
            paramKeyList={[]}
            compareRunCharts={uiState.compareRunCharts}
            compareRunSections={uiState.compareRunSections}
            groupBy={groupBy}
            hideEmptyCharts={uiState.hideEmptyCharts}
            chartsSearchFilter={uiState.chartsSearchFilter}
            storageKey="some-experiment-id"
            minWidth={800}
          />
        </ExperimentPageUIStateContextProvider>
      );
    };
    return renderWithIntl(
      <DesignSystemProvider>
        <MockedReduxStoreProvider
          state={{
            entities: {
              paramsByRunUuid: {},
              latestMetricsByRunUuid,
              tagsByRunUuid: {},
              imagesByRunUuid,
              colorByRunUuid: {},
            },
          }}
        >
          <TestComponent />
        </MockedReduxStoreProvider>
      </DesignSystemProvider>,
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

  test('drag and drop (reorder) across sections', async () => {
    createComponentMock({
      // Let's have at least one run in the comparison
      comparedRuns: [{ runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } } as any],
    });

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'metric-beta' })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'tmp/metric-omega' })).toBeInTheDocument();
    });

    // Find all sections (draggable sub-grids)
    const allSections = screen.getAllByTestId('draggable-chart-cards-grid');
    // Locate "tmp/" section and "metric-beta" chart element
    const tmpGridArea = allSections.find((section) => section.textContent?.includes('tmp/metric-omega')) as HTMLElement;

    const betaMetricChartElement = getChartArea('metric-beta');
    const betaMetricChartHandle = within(betaMetricChartElement).getByTestId(
      'experiment-view-compare-runs-card-drag-handle',
    );

    // Drag "beta" chart into the "tmp" chart area which is occupied by "omega"
    fireEvent.mouseDown(betaMetricChartHandle);
    fireEvent.mouseMove(tmpGridArea);
    fireEvent.mouseUp(betaMetricChartHandle);

    const betaChartEntry = currentUIState.compareRunCharts?.find(({ uuid }) => uuid === 'metric-beta');
    const omegaChartEntry = currentUIState.compareRunCharts?.find(({ uuid }) => uuid === 'metric-omega');

    await waitFor(() => {
      expect(betaChartEntry?.metricSectionId).toBe(omegaChartEntry?.metricSectionId);
    });
  });

  test('initializes correct chart types for given initial runs data', async () => {
    currentUIState.compareRunCharts = undefined;

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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for metric-static]'),
        screen.getByText('[line plot for metric-with-history]'),
      ]);
    });
  });

  test('initializes sections', async () => {
    currentUIState.compareRunCharts = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: {} },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: {} },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });
  });

  test('drag and drop sections', async () => {
    createComponentMock({
      // Let's have at least one run in the comparison
      comparedRuns: [{ runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } } as any],
    });

    await waitFor(() => {
      expect(screen.getByText('tmp')).toBeInTheDocument();
      expect(screen.getByText('Model metrics')).toBeInTheDocument();
      expect(screen.getByText('System metrics')).toBeInTheDocument();
    });

    const metricSection0 = getSectionArea('tmp');
    const metricSection1 = getSectionArea('Model metrics');
    const metricSection2 = getSectionArea('System metrics');
    const metricSection0Handle = within(metricSection0).getByTestId(
      'experiment-view-compare-runs-section-header-drag-handle',
    );

    // Move section 'tmp' to section 'System metrics'
    act(() => {
      fireEvent.dragStart(metricSection0Handle);
      fireEvent.dragEnter(metricSection2);
      fireEvent.dragOver(metricSection2);
      fireEvent.drop(metricSection2);
    });

    await waitFor(async () => {
      assertElementsInOrder([
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
        screen.getByText('tmp'),
      ]);
    });

    const metricSection2Handle = within(metricSection2).getByTestId(
      'experiment-view-compare-runs-section-header-drag-handle',
    );

    // Move section 'System metrics' to section 'Model metrics'
    act(() => {
      fireEvent.dragStart(metricSection2Handle);
      fireEvent.dragEnter(metricSection1);
      fireEvent.dragOver(metricSection1);
      fireEvent.drop(metricSection1);
    });

    await waitFor(async () => {
      assertElementsInOrder([
        screen.getByText('System metrics'),
        screen.getByText('Model metrics'),
        screen.getByText('tmp'),
      ]);
    });
  });

  test('detecting and adding new sections when already in default order', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
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

    await waitFor(() => {
      expect(screen.queryByText('tmp1')).not.toBeInTheDocument();
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('tmp2'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });

    expect(currentUIState.isAccordionReordered).toBe(false);

    cleanup();

    // New run is added to the end of the list

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
        { runUuid: 'run_new', runName: 'New run', runInfo: { runUuid: 'run_new' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('tmp1'),
        screen.getByText('tmp2'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });
  });

  test('detecting and adding new sections when not in default order', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('tmp2'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });

    // Update order of compareRunSections
    const tmpSection = getSectionArea('tmp');
    const tmp2Section = getSectionArea('tmp2');
    const tmpSectionHandle = within(tmpSection).getByTestId('experiment-view-compare-runs-section-header-drag-handle');
    // Move section 'tmp' to section 'tmp2'
    act(() => {
      fireEvent.dragStart(tmpSectionHandle);
      fireEvent.dragEnter(tmp2Section);
      fireEvent.dragOver(tmp2Section);
      fireEvent.drop(tmp2Section);
    });

    expect((await getCurrentUIState()).isAccordionReordered).toBe(true);

    cleanup();

    // New run is added to the end of the list
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
        { runUuid: 'run_new', runName: 'New run', runInfo: { runUuid: 'run_new' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp2'),
        screen.getByText('tmp'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
        screen.getByText('tmp1'),
      ]);
    });
  });

  test('detecting metric history and updating bar chart to line chart', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    // Run with static metric
    createComponentMock({
      comparedRuns: [{ runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } }] as any,
      latestMetricsByRunUuid: {
        run_1: {
          'tmp/metric-with-history': { key: 'tmp/metric-with-history', value: 1, step: 0 },
        },
      } as any,
    });

    await waitFor(() => {
      // Test that the chart is initialized as a bar chart
      expect(screen.getByText('[bar plot for tmp/metric-with-history]')).toBeInTheDocument();
    });

    cleanup();

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
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

    await waitFor(() => {
      // Test that the chart is updated to a line chart
      expect(screen.getByText('[line plot for tmp/metric-with-history]')).toBeInTheDocument();
    });
  });

  test('detecting new run and inserting new chart when section is not reordered', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section1/metric3]'),
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section1/metric6]'),
        screen.getByText('[bar plot for section1/metric7]'),
      ]);
    });

    cleanup();

    // Inserting a new run with section1/metric4
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
        { runUuid: 'run_3', runName: 'Third run', runInfo: { runUuid: 'run_3' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section1/metric3]'),
        screen.getByText('[bar plot for section1/metric4]'),
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section1/metric6]'),
        screen.getByText('[bar plot for section1/metric7]'),
      ]);
    });
  });

  test('detecting new run and inserting new chart when section is reordered by dragging chart out', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section2/metric6]'),
      ]);
    });

    // Drag section1/metric5 into section2
    const chartArea5 = getChartArea('section1/metric5');
    const chartArea6 = getChartArea('section2/metric6');

    // Find all sections (draggable sub-grids)
    const allSections = screen.getAllByTestId('draggable-chart-cards-grid');
    // Locate "section2/" section and "metric-5" chart element
    const section2Area = allSections.find((grid) => grid.textContent?.includes('section2/metric6')) as HTMLElement;

    // Get section1/metric5 handle
    const chartArea5Handle = within(chartArea5).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag section1/metric5 chart into section2/metric6 chart position
    fireEvent.mouseDown(chartArea5Handle);
    fireEvent.mouseMove(section2Area);
    fireEvent.mouseUp(chartArea5Handle);

    cleanup();

    // Adding a new run with section1/metric4 should append the metric at the end
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
        { runUuid: 'run_3', runName: 'Third run', runInfo: { runUuid: 'run_3' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section1/metric3]'),
        screen.getByText('[bar plot for section1/metric7]'),
        screen.getByText('[bar plot for section1/metric4]'),
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section2/metric6]'),
      ]);
    });
  });

  test('detecting new run and inserting new chart when section is reordered by dragging chart in', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section2/metric6]'),
      ]);
    });

    // Drag section2/metric6 into section1
    const chartArea6 = getChartArea('section2/metric6');
    const chartArea5 = getChartArea('section1/metric5');

    // Find all sections (draggable sub-grids)
    const allSections = screen.getAllByTestId('draggable-chart-cards-grid');
    // Locate "section1/" section and "metric-6" chart element
    const section1Area = allSections.find((grid) => grid.textContent?.includes('section1/metric5')) as HTMLElement;

    // Get section2/metric6 handle
    const chartArea6Handle = within(chartArea6).getByTestId('experiment-view-compare-runs-card-drag-handle');

    // Drag "section2/metric6" chart into the "section1/metric5" chart position
    fireEvent.mouseDown(chartArea6Handle);
    fireEvent.mouseMove(section1Area);
    fireEvent.mouseUp(chartArea6Handle);

    cleanup();

    // Adding a new run with section1/metric4 should append the metric at the end
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_1', runName: 'First run', runInfo: { runUuid: 'run_1' } },
        { runUuid: 'run_2', runName: 'Second run', runInfo: { runUuid: 'run_2' } },
        { runUuid: 'run_3', runName: 'Third run', runInfo: { runUuid: 'run_3' } },
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

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('[bar plot for section2/metric6]'),
        screen.getByText('[bar plot for section1/metric3]'),
        screen.getByText('[bar plot for section1/metric5]'),
        screen.getByText('[bar plot for section1/metric7]'),
        screen.getByText('[bar plot for section1/metric4]'),
      ]);
    });
  });

  test('detecting new run and inserting new section when image is logged', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
      ] as any,
      imagesByRunUuid: {
        run_latest: {
          'tmp/image': {
            'tmp/image.png': {
              filepath: 'tmp/image.png',
            },
          },
        },
        run_oldest: {
          'tmp2/image1': {
            'tmp2/image1.png': {
              filepath: 'images/tmp2/image1.png',
            },
          },
        },
      } as any,
    });

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('tmp2'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });

    expect((await getCurrentUIState()).isAccordionReordered).toBe(false);

    cleanup();

    // New run is added to the end of the list
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
        { runUuid: 'run_new', runName: 'New run', runInfo: { runUuid: 'run_new' } },
      ] as any,
      imagesByRunUuid: {
        run_latest: {
          'tmp/image': {
            'tmp/image.png': {
              filepath: 'tmp/image.png',
            },
          },
        },
        run_oldest: {
          'tmp2/image1': {
            'tmp2/image1.png': {
              filepath: 'images/tmp2/image1.png',
            },
          },
        },
        run_new: {
          'tmp1/image3': {
            'tmp1/image3.png': {
              filepath: 'images/tmp1/image3.png',
            },
          },
        },
      } as any,
    });

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp'),
        screen.getByText('tmp1'),
        screen.getByText('tmp2'),
        screen.getByText('Model metrics'),
        screen.getByText('System metrics'),
      ]);
    });
  });

  test('detecting new run and inserting new chart when image is logged', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
      ] as any,
      imagesByRunUuid: {
        run_latest: {
          'tmp/image': {
            'tmp/image.png': {
              filepath: 'tmp/image.png',
            },
          },
        },
        run_oldest: {
          'tmp2/image1': {
            'tmp2/image1.png': {
              filepath: 'images/tmp2/image1.png',
            },
          },
        },
      } as any,
    });

    // Rerender with compareRunCharts and compareRunSections initialized
    act(() => {
      cleanup();
    });
    createComponentMock();

    // Rerender with compareRunCharts and compareRunSections initialized
    act(() => {
      cleanup();
    });

    // New run is added to the end of the list
    createComponentMock({
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: { runUuid: 'run_oldest' } },
        { runUuid: 'run_new', runName: 'New run', runInfo: { runUuid: 'run_new' } },
      ] as any,
      imagesByRunUuid: {
        run_latest: {
          'tmp/image': {
            'tmp/image.png': {
              filepath: 'tmp/image.png',
            },
          },
        },
        run_oldest: {
          'tmp2/image1': {
            'tmp2/image1.png': {
              filepath: 'images/tmp2/image1.png',
            },
          },
        },
        run_new: {
          'tmp1/image3': {
            'tmp1/image3.png': {
              filepath: 'images/tmp1/image3.png',
            },
          },
        },
      } as any,
    });

    await waitFor(() => {
      assertElementsInOrder([
        screen.getByText('tmp/image'),
        screen.getByText('tmp1/image3'),
        screen.getByText('tmp2/image1'),
      ]);
    });
  });

  test('search functionality filters metric charts', async () => {
    jest.useFakeTimers();
    const customUserEvent = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    // For this test, add chart with multiple metrics at the end
    currentUIState.compareRunCharts = [...testCharts, testMultipleMetricsLineChart];

    createComponentMock({
      // Let's have at least one run in the comparison
      comparedRuns: [{ runUuid: 'run_latest', runName: 'Last run', runInfo: { runUuid: 'run_latest' } } as any],
    });

    await waitFor(() => {
      expect(screen.queryByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: 'metric-beta' })).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: 'metric-gamma' })).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).toBeInTheDocument();
      expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).toBeInTheDocument();
    });

    // Type search query into searchbox
    await customUserEvent.type(screen.getByRole('searchbox'), 'alpha');

    // Wait for debounce to kick in
    act(() => {
      jest.advanceTimersByTime(500);
    });

    const modelMetricsSection = screen.getByText('Model metrics');
    const sectionHeader = modelMetricsSection.closest(
      '[data-testid="experiment-view-compare-runs-section-header"]',
    ) as HTMLElement;

    // Expect particular charts to disappear
    expect(screen.queryByRole('heading', { name: 'metric-beta' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).toBeInTheDocument();

    // Expect metric section to be displayed
    expect(
      (await getCurrentUIState()).compareRunSections?.find(({ name }) => name === 'Model metrics')?.display,
    ).toEqual(true);

    // Click on the section header
    await customUserEvent.click(sectionHeader);

    // Expect metric section to be hidden
    expect(
      (await getCurrentUIState()).compareRunSections?.find(({ name }) => name === 'Model metrics')?.display,
    ).toEqual(false);

    // Put a regexp that matches 'tmp/'-prefixed and 'gamma'-suffixed metrics
    await customUserEvent.clear(screen.getByRole('searchbox'));
    await customUserEvent.type(screen.getByRole('searchbox'), '(tmp/.+)|gamma');

    // Wait for debounce to kick in
    act(() => {
      jest.advanceTimersByTime(500);
    });

    // Expect 'tmp/'-prefixed and 'gamma'-suffixed metrics to be displayed
    expect(screen.queryByRole('heading', { name: 'metric-beta' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).not.toBeInTheDocument();

    // Put a regexp that matches all metrics starting with 'metric-' but not 'gamma'-suffixed.
    // Try to include some uppercase letters.
    await customUserEvent.clear(screen.getByRole('searchbox'));
    await customUserEvent.type(screen.getByRole('searchbox'), '^mETRIc-(?!gamma).*');

    // Wait for debounce to kick in
    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(screen.queryByRole('heading', { name: 'metric-beta' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).toBeInTheDocument();

    // Put a regexp that doesn't match any metrics
    await customUserEvent.clear(screen.getByRole('searchbox'));
    await customUserEvent.type(screen.getByRole('searchbox'), 'yotta');

    // Wait for debounce to kick in
    act(() => {
      jest.advanceTimersByTime(500);
    });

    // Expect all charts to be hidden
    expect(screen.queryByRole('heading', { name: 'metric-beta' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-gamma' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'tmp/metric-omega' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-alpha' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'metric-beta vs metric-alpha' })).not.toBeInTheDocument();

    jest.useRealTimers();
  });

  test('correctly determines chart types run groups mixed with ungrouped runs', async () => {
    currentUIState.compareRunCharts = undefined;
    currentUIState.compareRunSections = undefined;

    // Render a component with a group and a run:
    // - a group contains aggregated "metric_1" data indicating history
    // - a group contains aggregated "metric_2" data not indicating any history
    // - a run contains "metric_3" data indicating history
    createComponentMock({
      groupBy: 'metric_1',
      comparedRuns: [
        {
          runUuid: '',
          groupParentInfo: {
            groupId: 'some_group',
            groupingValues: [],
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
          runInfo: { runUuid: 'run_1' },
          runDateAndNestInfo: { belongsToGroup: false },
        },
      ] as any,
      latestMetricsByRunUuid: {
        run_1: {
          metric_3: { key: 'metric_3', value: 1, step: 3 },
        },
      } as any,
    });

    await waitFor(() => {
      // Assert correct chart types for metrics
      assertElementsInOrder([
        screen.getByText('[line plot for metric_1]'),
        screen.getByText('[bar plot for metric_2]'),
        screen.getByText('[line plot for metric_3]'),
      ]);
    });
  });

  describe('hiding charts with no data', () => {
    // Set up charts
    beforeEach(() => {
      currentUIState.compareRunCharts = [
        {
          type: RunsChartType.BAR,
          uuid: 'chart-alpha',
          runsCountToCompare: 10,
          metricKey: 'metric-alpha',
          metricSectionId: 'metric-section-1',
          deleted: false,
          isGenerated: true,
        } as RunsChartsBarCardConfig,
        {
          type: RunsChartType.PARALLEL,
          uuid: 'chart-parallel',
          runsCountToCompare: 10,
          metricSectionId: 'metric-section-1',
          selectedMetrics: ['metric-parallel-1', 'metric-parallel-2'],
          deleted: false,
          isGenerated: true,
        } as RunsChartsParallelCardConfig,
        {
          type: RunsChartType.DIFFERENCE,
          uuid: 'chart-difference',
          runsCountToCompare: 10,
          metricSectionId: 'metric-section-1',
          compareGroups: [DifferenceCardConfigCompareGroup.MODEL_METRICS],
          chartName: 'Runs difference view',
          showDifferencesOnly: true,
          deleted: false,
          isGenerated: true,
        } as RunsChartsDifferenceCardConfig,
      ];
    });

    // Set up non-configured charts
    const noConfigCharts = [
      {
        type: RunsChartType.PARALLEL,
        uuid: 'chart-parallel',
        runsCountToCompare: 10,
        metricSectionId: 'metric-section-1',
        selectedMetrics: [] as string[],
        deleted: false,
        isGenerated: true,
      } as RunsChartsParallelCardConfig,
      {
        type: RunsChartType.DIFFERENCE,
        uuid: 'chart-difference',
        runsCountToCompare: 10,
        metricSectionId: 'metric-section-1',
        compareGroups: [] as DifferenceCardConfigCompareGroup[],
        chartName: 'Runs difference view',
        showDifferencesOnly: true,
        deleted: false,
        isGenerated: true,
      } as RunsChartsDifferenceCardConfig,
    ];

    // Set up metrics store to contain data for unrelated metric
    const componentProps = {
      latestMetricsByRunUuid: {
        run_latest: {
          'metric-unrelated': { key: 'metric-unrelated', value: 1, step: 0 },
        },
        run_oldest: {
          'metric-unrelated': { key: 'metric-unrelated', value: 1, step: 0 },
        },
      } as any,
      comparedRuns: [
        { runUuid: 'run_latest', runName: 'Last run', runInfo: {} },
        { runUuid: 'run_oldest', runName: 'First run', runInfo: {} },
      ] as any,
    };

    test('displays a warning when configured charts do not contain corresponding data and hiding empty charts is disabled', async () => {
      currentUIState.hideEmptyCharts = false;
      createComponentMock(componentProps);

      await waitFor(() => {
        expect(screen.getByText(/metric-alpha/)).toBeInTheDocument();
        expect(screen.getByText(/Parallel Coordinates/)).toBeInTheDocument();
        expect(screen.getAllByText(/No chart data available for the currently visible runs/)).toHaveLength(2);
        expect(screen.getByText(/Runs difference view/)).toBeInTheDocument();
      });
    });

    test('displays a warning when all runs are hidden', async () => {
      currentUIState.hideEmptyCharts = false;
      createComponentMock({ ...componentProps, comparedRuns: [] });

      await waitFor(() => {
        expect(screen.queryByText(/metric-alpha/)).not.toBeInTheDocument();
        expect(screen.getByText(/All runs are hidden\./)).toBeInTheDocument();
      });
    });

    test('does not display empty chart at all when hiding empty charts is set', async () => {
      currentUIState.hideEmptyCharts = true;
      createComponentMock(componentProps);

      await waitFor(() => {
        expect(screen.queryByText(/metric-alpha/)).not.toBeInTheDocument();
        expect(screen.queryByText(/Parallel Coordinates/)).not.toBeInTheDocument();
        expect(screen.queryByText(/No chart data available for the currently visible runs/)).not.toBeInTheDocument();

        // Runs difference view is visible even if hide empty charts is set
        expect(screen.getByText(/Runs difference view/)).toBeInTheDocument();
      });
    });

    test('does not display parallel coords or runs difference chart when not configured and hiding empty charts is set', async () => {
      currentUIState.hideEmptyCharts = true;
      currentUIState.compareRunCharts = noConfigCharts;
      createComponentMock(componentProps);

      await waitFor(() => {
        expect(screen.queryByText(/Parallel Coordinates/)).not.toBeInTheDocument();
        expect(screen.queryByText(/No chart data available for the currently visible runs/)).not.toBeInTheDocument();

        expect(screen.queryByText(/Runs difference view/)).not.toBeInTheDocument();
        expect(screen.queryByText(/No run differences to display/)).not.toBeInTheDocument();
      });
    });

    test('displays a warning when parallel coords or runs difference chart not configured and hiding empty charts is disabled', async () => {
      currentUIState.hideEmptyCharts = false;
      currentUIState.compareRunCharts = noConfigCharts;
      createComponentMock(componentProps);

      await waitFor(() => {
        expect(screen.getByText(/Parallel Coordinates/)).toBeInTheDocument();
        expect(screen.getByText(/Runs difference view/)).toBeInTheDocument();
        expect(screen.getAllByText(/Configure chart/)).toHaveLength(2);
      });
    });
  });
});

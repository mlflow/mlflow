import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { RunsChartsDraggableCardsGridSection } from './RunsChartsDraggableCardsGridSection';
import { noop } from 'lodash';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import type {
  RunsChartsBarCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsScatterCardConfig,
} from '../runs-charts.types';
import { RunsChartType } from '../runs-charts.types';
import { RunsChartsTooltipWrapper } from '../hooks/useRunsChartsTooltip';
import { useCallback, useState } from 'react';
import type {
  ExperimentPageUIState,
  ExperimentRunsChartsUIConfiguration,
} from '../../experiment-page/models/ExperimentPageUIState';
import { createExperimentPageUIState } from '../../experiment-page/models/ExperimentPageUIState';
import type { RunsChartsUIConfigurationSetter } from '../hooks/useRunsChartsUIConfiguration';
import { RunsChartsUIConfigurationContextProvider } from '../hooks/useRunsChartsUIConfiguration';
import { RunsChartsDraggableCardsGridContextProvider } from './RunsChartsDraggableCardsGridContext';
import type { ChartSectionConfig } from '../../../types';
import { Checkbox, DesignSystemProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableHidingChartsWithNoData: jest.fn(() => true),
}));

// Mock useIsInViewport hook to simulate that the chart element is in the viewport
jest.mock('../hooks/useIsInViewport', () => ({
  useIsInViewport: () => ({ isInViewport: true, setElementRef: jest.fn() }),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // Larger timeout for integration testing (drag and drop simlation)

describe('RunsChartsDraggableCardsGrid', () => {
  const renderTestComponent = (element: React.ReactElement) => {
    const noopTooltipComponent = () => <div />;
    render(element, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <RunsChartsTooltipWrapper component={noopTooltipComponent} contextData={{}}>
              <TestApolloProvider>
                <MockedReduxStoreProvider state={{ entities: { sampledMetricsByRunUuid: {} } }}>
                  {children}
                </MockedReduxStoreProvider>
              </TestApolloProvider>
            </RunsChartsTooltipWrapper>
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  const mockGridElementSize = (element: Element, width: number, height: number) => {
    element.getBoundingClientRect = jest.fn<any, any>(() => ({
      top: 0,
      left: 0,
      width,
      height,
    }));
  };

  test('drag and drop cards within a single section with 3 columns', async () => {
    const cards = [
      { type: RunsChartType.BAR, metricKey: 'metric_1', uuid: 'card_1' },
      { type: RunsChartType.BAR, metricKey: 'metric_2', uuid: 'card_2' },
      { type: RunsChartType.BAR, metricKey: 'metric_3', uuid: 'card_3' },
    ] as RunsChartsBarCardConfig[];

    const TestComponent = () => {
      const [uiState, setUIState] = useState<ExperimentRunsChartsUIConfiguration>({
        ...createExperimentPageUIState(),
        compareRunCharts: cards,
      });

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={setUIState}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={cards}>
            <RunsChartsDraggableCardsGridSection
              cardsConfig={uiState.compareRunCharts ?? []}
              chartRunData={[]}
              sectionId="abc"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              sectionConfig={{
                display: true,
                isReordered: false,
                name: 'section_1',
                uuid: 'section_1',
                cardHeight: 360,
                columns: 3,
              }}
            />
          </RunsChartsDraggableCardsGridContextProvider>
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    renderTestComponent(<TestComponent />);

    await waitFor(() => {
      expect(screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')).toHaveLength(3);

      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_1',
        'metric_2',
        'metric_3',
      ]);
    });

    mockGridElementSize(screen.getByTestId('draggable-chart-cards-grid'), 900, 600);

    let dragHandle = screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')[0];

    fireEvent.mouseDown(dragHandle);
    fireEvent.mouseMove(screen.getByTestId('draggable-chart-cards-grid'), { clientX: 850, clientY: 100 });
    fireEvent.mouseUp(dragHandle);

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_2',
        'metric_3',
        'metric_1',
      ]);
    });

    dragHandle = screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')[1];

    fireEvent.mouseDown(dragHandle);
    fireEvent.mouseMove(screen.getByTestId('draggable-chart-cards-grid'), { clientX: 100, clientY: 100 });
    fireEvent.mouseUp(dragHandle);

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_3',
        'metric_2',
        'metric_1',
      ]);
    });

    dragHandle = screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')[0];

    fireEvent.mouseDown(dragHandle);
    fireEvent.mouseMove(screen.getByTestId('draggable-chart-cards-grid'), { clientX: 850, clientY: 450 });
    fireEvent.mouseUp(dragHandle);

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_2',
        'metric_1',
        'metric_3',
      ]);
    });
  });

  test('reorder cards using move to top and bottom menu actions', async () => {
    const cards = [
      { type: RunsChartType.BAR, metricKey: 'metric_1', uuid: 'card_1' },
      { type: RunsChartType.BAR, metricKey: 'metric_2', uuid: 'card_2' },
      { type: RunsChartType.BAR, metricKey: 'metric_3', uuid: 'card_3' },
    ] as RunsChartsBarCardConfig[];

    const TestComponent = () => {
      const [uiState, setUIState] = useState<ExperimentRunsChartsUIConfiguration>({
        ...createExperimentPageUIState(),
        compareRunCharts: cards,
      });

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={setUIState}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={cards}>
            <RunsChartsDraggableCardsGridSection
              cardsConfig={uiState.compareRunCharts ?? []}
              chartRunData={[]}
              sectionId="abc"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              sectionConfig={{
                display: true,
                isReordered: false,
                name: 'section_1',
                uuid: 'section_1',
                cardHeight: 360,
                columns: 3,
              }}
            />
          </RunsChartsDraggableCardsGridContextProvider>
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    renderTestComponent(<TestComponent />);

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_1',
        'metric_2',
        'metric_3',
      ]);
    });

    await userEvent.click(screen.getAllByTestId('experiment-view-compare-runs-card-menu')[1]);
    await userEvent.click(screen.getByTestId('experiment-view-compare-runs-move-to-top'));

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_2',
        'metric_1',
        'metric_3',
      ]);
    });

    await userEvent.click(screen.getAllByTestId('experiment-view-compare-runs-card-menu')[0]);
    await userEvent.click(screen.getByTestId('experiment-view-compare-runs-move-to-bottom'));

    await waitFor(() => {
      expect(screen.getAllByRole('heading').map((element) => element.textContent)).toEqual([
        'metric_1',
        'metric_3',
        'metric_2',
      ]);
    });
  });

  test('resize charts and change column count', async () => {
    const cards = [
      { type: RunsChartType.BAR, metricKey: 'metric_1', uuid: 'card_1' },
      { type: RunsChartType.BAR, metricKey: 'metric_2', uuid: 'card_2' },
      { type: RunsChartType.BAR, metricKey: 'metric_3', uuid: 'card_3' },
    ] as RunsChartsBarCardConfig[];

    const TestComponent = () => {
      const initialState: ExperimentRunsChartsUIConfiguration = {
        ...createExperimentPageUIState(),
        compareRunCharts: cards,
        compareRunSections: [
          { uuid: 'section_a', display: true, isReordered: false, name: 'section_a', cardHeight: 300, columns: 3 },
        ],
      };
      const [uiState, setUIState] = useState<ExperimentRunsChartsUIConfiguration>(initialState);

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={setUIState}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={cards}>
            <RunsChartsDraggableCardsGridSection
              cardsConfig={uiState.compareRunCharts ?? []}
              chartRunData={[]}
              sectionId="section_a"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              sectionConfig={uiState.compareRunSections?.[0] as ChartSectionConfig}
            />
          </RunsChartsDraggableCardsGridContextProvider>
          <div>Columns displayed: {uiState.compareRunSections?.[0].columns}</div>
          <div>Card height: {uiState.compareRunSections?.[0].cardHeight}</div>
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    renderTestComponent(<TestComponent />);

    await waitFor(() => {
      expect(screen.getAllByTestId('draggable-card-resize-handle')).toHaveLength(3);
      expect(screen.getByText('Columns displayed: 3')).toBeInTheDocument();
      expect(screen.getByText('Card height: 300')).toBeInTheDocument();
    });

    mockGridElementSize(screen.getByTestId('draggable-chart-cards-grid'), 900, 600);

    const resizeHandle = screen.getAllByTestId('draggable-card-resize-handle')[0];

    fireEvent.mouseDown(resizeHandle);
    fireEvent.mouseMove(screen.getByTestId('draggable-chart-cards-grid'), { clientX: 899, clientY: 500 });
    fireEvent.mouseUp(resizeHandle);

    await waitFor(() => {
      expect(screen.getByText('Columns displayed: 1')).toBeInTheDocument();
      expect(screen.getByText('Card height: 500')).toBeInTheDocument();
    });

    fireEvent.mouseDown(resizeHandle);
    fireEvent.mouseMove(screen.getByTestId('draggable-chart-cards-grid'), { clientX: 460, clientY: 500 });
    fireEvent.mouseUp(resizeHandle);

    await waitFor(() => {
      expect(screen.getByText('Columns displayed: 2')).toBeInTheDocument();
      expect(screen.getByText('Card height: 500')).toBeInTheDocument();
    });
  });

  test('drag and drop cards between two sections', async () => {
    const cards = [
      { type: RunsChartType.BAR, metricKey: 'metric_1', uuid: 'card_a_1', metricSectionId: 'section_a' },
      { type: RunsChartType.BAR, metricKey: 'metric_2', uuid: 'card_a_2', metricSectionId: 'section_a' },
      { type: RunsChartType.BAR, metricKey: 'metric_3', uuid: 'card_a_3', metricSectionId: 'section_a' },

      { type: RunsChartType.BAR, metricKey: 'metric_4', uuid: 'card_b_1', metricSectionId: 'section_b' },
      { type: RunsChartType.BAR, metricKey: 'metric_5', uuid: 'card_b_2', metricSectionId: 'section_b' },
    ] as RunsChartsBarCardConfig[];

    const TestComponent = () => {
      const [firstGridSection, secondGridSection] = [
        {
          display: true,
          isReordered: false,
          name: 'section_a',
          uuid: 'section_a',
          cardHeight: 300,
          columns: 3,
        },
        {
          display: true,
          isReordered: false,
          name: 'section_b',
          uuid: 'section_b',
          cardHeight: 300,
          columns: 2,
        },
      ];
      const [uiState, setUIState] = useState<ExperimentRunsChartsUIConfiguration>({
        ...createExperimentPageUIState(),
        compareRunCharts: cards,
        compareRunSections: [firstGridSection, secondGridSection],
      });

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={setUIState}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={cards}>
            <RunsChartsDraggableCardsGridSection
              cardsConfig={
                uiState.compareRunCharts?.filter(({ metricSectionId }) => metricSectionId === 'section_a') ?? []
              }
              chartRunData={[]}
              sectionId="section_a"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              sectionConfig={firstGridSection}
            />
            <RunsChartsDraggableCardsGridSection
              cardsConfig={
                uiState.compareRunCharts?.filter(({ metricSectionId }) => metricSectionId === 'section_b') ?? []
              }
              chartRunData={[]}
              sectionId="section_b"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              sectionConfig={secondGridSection}
            />
          </RunsChartsDraggableCardsGridContextProvider>
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    renderTestComponent(<TestComponent />);

    await waitFor(() => {
      expect(screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')).toHaveLength(5);
    });

    const [firstGrid, secondGrid] = screen.getAllByTestId('draggable-chart-cards-grid');

    await waitFor(() => {
      expect(
        within(firstGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_1', 'metric_2', 'metric_3']);

      expect(
        within(secondGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_4', 'metric_5']);
    });

    mockGridElementSize(firstGrid, 900, 600);
    mockGridElementSize(secondGrid, 900, 600);

    let dragHandle = within(secondGrid).getAllByTestId('experiment-view-compare-runs-card-drag-handle')[0];

    fireEvent.mouseDown(dragHandle);
    fireEvent.mouseMove(firstGrid, { clientX: 450, clientY: 100 });
    fireEvent.mouseUp(dragHandle);

    await waitFor(() => {
      expect(
        within(firstGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_1', 'metric_4', 'metric_2', 'metric_3']);

      expect(
        within(secondGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_5']);
    });

    dragHandle = within(firstGrid).getAllByTestId('experiment-view-compare-runs-card-drag-handle')[2];

    fireEvent.mouseDown(dragHandle);
    fireEvent.mouseMove(secondGrid, { clientX: 400, clientY: 100 });
    fireEvent.mouseUp(dragHandle);

    await waitFor(() => {
      expect(
        within(firstGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_1', 'metric_4', 'metric_3']);

      expect(
        within(secondGrid)
          .getAllByRole('heading')
          .map((element) => element.textContent),
      ).toEqual(['metric_2', 'metric_5']);
    });
  });

  test('properly hide cards with no data', async () => {
    const cards = [
      { type: RunsChartType.BAR, metricKey: 'metric_1', uuid: 'card_1' } as RunsChartsBarCardConfig,
      { type: RunsChartType.LINE, selectedMetricKeys: ['metric_1'], uuid: 'card_2' } as RunsChartsLineCardConfig,
      {
        type: RunsChartType.SCATTER,
        xaxis: { key: 'metric_1', type: 'METRIC' },
        yaxis: { key: 'metric_2', type: 'METRIC' },
        uuid: 'card_3',
      } as RunsChartsScatterCardConfig,
      {
        type: RunsChartType.CONTOUR,
        xaxis: { key: 'metric_1', type: 'METRIC' },
        yaxis: { key: 'metric_2', type: 'METRIC' },
        zaxis: { key: 'metric_3', type: 'METRIC' },
        uuid: 'card_4',
      } as RunsChartsContourCardConfig,
      {
        type: RunsChartType.PARALLEL,
        selectedMetrics: ['metric_1', 'metric_2', 'metric_3'],
        uuid: 'card_5',
      } as RunsChartsParallelCardConfig,
    ];

    const TestComponent = () => {
      const [uiState, updateUIState] = useState<ExperimentPageUIState>({
        ...createExperimentPageUIState(),
        hideEmptyCharts: false,
        compareRunCharts: cards,
      });

      const updateChartsUIState = useCallback<(stateSetter: RunsChartsUIConfigurationSetter) => void>(
        (setter) => {
          updateUIState((state) => ({
            ...state,
            ...setter(state),
          }));
        },
        [updateUIState],
      );

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={updateChartsUIState}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={cards}>
            <RunsChartsDraggableCardsGridSection
              cardsConfig={uiState.compareRunCharts ?? []}
              chartRunData={[
                {
                  displayName: 'run_1',
                  // Metrics unrelated to the cards
                  metrics: { metric_4: [{ key: 'metric_4', value: 1 }] },
                } as any,
              ]}
              sectionId="test"
              setFullScreenChart={noop}
              groupBy={null}
              onRemoveChart={noop}
              onStartEditChart={noop}
              hideEmptyCharts={uiState.hideEmptyCharts}
              sectionConfig={{} as any}
            />
            <Checkbox
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsdraggablecardsgrid.test.tsx_420"
              isChecked={uiState.hideEmptyCharts}
              onChange={(checked) => {
                updateChartsUIState((state) => ({ ...state, hideEmptyCharts: checked }));
              }}
            >
              Hide empty charts
            </Checkbox>
          </RunsChartsDraggableCardsGridContextProvider>
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    renderTestComponent(<TestComponent />);

    await waitFor(() => {
      expect(screen.getAllByTestId('experiment-view-compare-runs-card-drag-handle')).toHaveLength(5);
    });

    await userEvent.click(screen.getByText('Hide empty charts'));

    await waitFor(() => {
      expect(screen.queryAllByTestId('experiment-view-compare-runs-card-drag-handle')).toHaveLength(0);
      expect(screen.getByText('No charts in this section')).toBeInTheDocument();
    });
  });
});

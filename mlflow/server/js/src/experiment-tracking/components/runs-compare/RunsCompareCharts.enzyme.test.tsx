import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { RunsChartsBarChartCard } from '../runs-charts/components/cards/RunsChartsBarChartCard';
import { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import {
  RunsChartsCardConfig,
  RunsChartType,
  RunsChartsContourCardConfig,
  RunsChartsScatterCardConfig,
  RunsChartsParallelCardConfig,
} from '../runs-charts/runs-charts.types';
import { RunsCompareCharts } from './RunsCompareCharts';
import { RunsChartsTooltipBodyComponent, RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { RunsChartsLineChartCard } from '../runs-charts/components/cards/RunsChartsLineChartCard';
import { RunsChartsScatterChartCard } from '../runs-charts/components/cards/RunsChartsScatterChartCard';
import { RunsChartsContourChartCard } from '../runs-charts/components/cards/RunsChartsContourChartCard';
import { RunsChartsParallelChartCard } from '../runs-charts/components/cards/RunsChartsParallelChartCard';
import {
  filterParallelCoordinateData,
  MAX_NUMBER_STRINGS,
  ParallelCoordinateDataEntry,
} from '../runs-charts/components/charts/LazyParallelCoordinatesPlot';

jest.mock('../runs-charts/components/cards/RunsChartsBarChartCard', () => ({
  RunsChartsBarChartCard: () => <div />,
}));

jest.mock('../runs-charts/components/cards/RunsChartsLineChartCard', () => ({
  RunsChartsLineChartCard: () => <div />,
}));

jest.mock('@emotion/react', () => ({
  ...(jest.requireActual('@emotion/react') as any),
  useTheme: () => ({ colors: {} }),
}));

describe('RunsCompareCharts', () => {
  let onEditChart: () => void;
  let onRemoveChart: () => void;

  beforeEach(() => {
    onEditChart = jest.fn();
    onRemoveChart = jest.fn();
  });

  const defaultBodyComponent: RunsChartsTooltipBodyComponent = ({ runUuid }) => (
    <div data-testid="tooltip-body">
      tooltip body
      <div data-testid="tooltip-body-run-uuid">{runUuid}</div>
    </div>
  );

  const createComponentMock = (
    cards: RunsChartsCardConfig[] = [],
    runs: RunsChartsRunData[] = [],
    contextData: string | undefined = undefined,
  ) =>
    mountWithIntl(
      <RunsChartsTooltipWrapper contextData={contextData} component={defaultBodyComponent}>
        <RunsCompareCharts
          chartRunData={runs}
          onRemoveChart={onRemoveChart}
          onStartEditChart={onEditChart}
          cardsConfig={cards}
          onReorderCharts={() => {}}
          groupBy=""
        />
      </RunsChartsTooltipWrapper>,
    );

  test('should not display chart components when there is no cards configured', () => {
    const wrapper = createComponentMock([]);
    expect(wrapper.find(RunsChartsBarChartCard).length).toBe(0);
  });

  test('should display chart components for bar charts', () => {
    const runs = [{ metrics: {}, params: {}, runInfo: { run_uuid: 'abc' } }];
    const wrapper = createComponentMock(
      [
        { type: RunsChartType.BAR, deleted: false, isGenerated: true },
        {
          type: RunsChartType.CONTOUR,
          xaxis: { key: '', type: 'METRIC' },
          yaxis: { key: '', type: 'METRIC' },
          zaxis: { key: '', type: 'METRIC' },
        } as RunsChartsContourCardConfig,
        { type: RunsChartType.LINE, deleted: false, isGenerated: true },
        {
          type: RunsChartType.SCATTER,
          xaxis: { key: '', type: 'METRIC' },
          yaxis: { key: '', type: 'METRIC' },
        } as RunsChartsScatterCardConfig,
        {
          type: RunsChartType.PARALLEL,
          selectedParams: [],
          selectedMetrics: [],
          deleted: false,
          isGenerated: true,
        } as RunsChartsParallelCardConfig,
        { type: RunsChartType.BAR, deleted: false, isGenerated: true },
      ],
      runs as RunsChartsRunData[],
    );

    // Expect two bar charts in the set
    expect(wrapper.find(RunsChartsBarChartCard).length).toBe(2);

    // Expect one line chart in the set
    expect(wrapper.find(RunsChartsLineChartCard).length).toBe(1);

    // Expect one scatter chart in the set
    expect(wrapper.find(RunsChartsScatterChartCard).length).toBe(1);

    // Expect one contour chart in the set
    expect(wrapper.find(RunsChartsContourChartCard).length).toBe(1);

    // Expect one parallel coordinates chart in the set
    expect(wrapper.find(RunsChartsParallelChartCard).length).toBe(1);

    const allChartComponents = [
      wrapper.find(RunsChartsBarChartCard),
      wrapper.find(RunsChartsLineChartCard),
      wrapper.find(RunsChartsScatterChartCard),
      wrapper.find(RunsChartsContourChartCard),
      wrapper.find(RunsChartsParallelChartCard),
    ];

    // Make sure that all elements are getting runs passed
    for (const collection of allChartComponents) {
      for (const chartInstance of collection.getElements()) {
        expect(chartInstance.props.chartRunData).toEqual(runs);
      }
    }
  });

  test('parallel coord chart filter out NaNs and nulls', () => {
    const data: ParallelCoordinateDataEntry[] = [];

    for (let i = 0; i < 100; i++) {
      data.push({
        uuid: i,
        left: Math.random(),
        right: Math.random(),
      });
    }
    data.push({
      uuid: 100,
      left: NaN,
      right: Math.random(),
    });
    data.push({
      uuid: 101,
      left: null,
      right: Math.random(),
    });
    expect(data.length).toBe(102);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe(100);
  });

  test('parallel coord chart only keep a max of 30 unique string values', () => {
    const data = [];
    const divisor = 2;
    for (let i = 0; i < 100; i++) {
      data.push({
        uuid: i,
        left: `${Math.floor(i / divisor)}a`,
        right: Math.random(),
      });
    }
    expect(data.length).toBe(100);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe(MAX_NUMBER_STRINGS * divisor);
  });

  test('parallel coord chart displays 100 nums over 50 strings', () => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      data.push({
        uuid: i,
        left: Math.random(),
        right: Math.random(),
      });
    }
    for (let i = 100; i < 150; i++) {
      data.push({
        uuid: i,
        left: `${Math.floor(i / 2)}a`,
        right: Math.random(),
      });
    }
    expect(data.length).toBe(150);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe(100);
  });

  test('parallel coord chart displays 90/99 strings over 51 nums', () => {
    const data = [];
    const divisor = 3;
    for (let i = 0; i < 51; i++) {
      data.push({
        uuid: i,
        left: Math.random(),
        right: Math.random(),
      });
    }
    for (let i = 51; i < 150; i++) {
      data.push({
        uuid: i,
        left: `${Math.floor(i / divisor)}a`,
        right: Math.random(),
      });
    }
    expect(data.length).toBe(150);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe(divisor * MAX_NUMBER_STRINGS);
  });

  test('parallel coord chart 3 column', () => {
    const data = [];
    const divisor = 4;
    for (let i = 0; i < 200; i++) {
      if (i % 4 === 0) {
        data.push({
          uuid: i,
          left: Math.random(),
          middle: 'a',
          right: Math.random(),
        });
      } else {
        data.push({
          uuid: i,
          left: `${Math.floor(i / divisor)}a`,
          middle: 'b',
          right: Math.random(),
        });
      }
    }

    expect(data.length).toBe(200);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe((divisor - 1) * MAX_NUMBER_STRINGS);
  });

  test('no values shown', () => {
    const data: any = [];

    expect(data.length).toBe(0);
    const filteredData = filterParallelCoordinateData(data);
    expect(filteredData.length).toBe(0);
  });
});

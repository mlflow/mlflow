import { mount } from 'enzyme';
import { RunsCompareBarChartCard } from './cards/RunsCompareBarChartCard';
import { CompareChartRunData } from './charts/CompareRunsCharts.common';
import {
  RunsCompareCardConfig,
  RunsCompareChartType,
  RunsCompareContourCardConfig,
  RunsCompareScatterCardConfig,
  RunsCompareParallelCardConfig,
} from './runs-compare.types';
import { RunsCompareCharts } from './RunsCompareCharts';
import {
  CompareRunsTooltipBodyComponent,
  CompareRunsTooltipWrapper,
} from './hooks/useCompareRunsTooltip';
import { RunsCompareLineChartCard } from './cards/RunsCompareLineChartCard';
import { RunsCompareScatterChartCard } from './cards/RunsCompareScatterChartCard';
import { RunsCompareContourChartCard } from './cards/RunsCompareContourChartCard';
import { RunsCompareParallelChartCard } from './cards/RunsCompareParallelChartCard';
import {
  filterParallelCoordinateData,
  MAX_NUMBER_STRINGS,
  ParallelCoordinateDataEntry,
} from './charts/LazyParallelCoordinatesPlot';

jest.mock('./cards/RunsCompareBarChartCard', () => ({
  RunsCompareBarChartCard: () => <div />,
}));

jest.mock('./cards/RunsCompareLineChartCard', () => ({
  RunsCompareLineChartCard: () => <div />,
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

  const defaultBodyComponent: CompareRunsTooltipBodyComponent = ({ runUuid }) => (
    <div data-testid='tooltip-body'>
      tooltip body
      <div data-testid='tooltip-body-run-uuid'>{runUuid}</div>
    </div>
  );

  const createComponentMock = (
    cards: RunsCompareCardConfig[] = [],
    runs: CompareChartRunData[] = [],
    contextData: string | undefined = undefined,
  ) =>
    mount(
      <CompareRunsTooltipWrapper contextData={contextData} component={defaultBodyComponent}>
        <RunsCompareCharts
          chartRunData={runs}
          onRemoveChart={onRemoveChart}
          onStartEditChart={onEditChart}
          cardsConfig={cards}
        />
      </CompareRunsTooltipWrapper>,
    );

  test('should not display chart components when there is no cards configured', () => {
    const wrapper = createComponentMock([]);
    expect(wrapper.find(RunsCompareBarChartCard).length).toBe(0);
  });

  test('should display chart components for bar charts', () => {
    const runs = [{ metrics: {}, params: {}, runInfo: { run_uuid: 'abc' } }];
    const wrapper = createComponentMock(
      [
        { type: RunsCompareChartType.BAR },
        {
          type: RunsCompareChartType.CONTOUR,
          xaxis: { key: '', type: 'METRIC' },
          yaxis: { key: '', type: 'METRIC' },
          zaxis: { key: '', type: 'METRIC' },
        } as RunsCompareContourCardConfig,
        { type: RunsCompareChartType.LINE },
        {
          type: RunsCompareChartType.SCATTER,
          xaxis: { key: '', type: 'METRIC' },
          yaxis: { key: '', type: 'METRIC' },
        } as RunsCompareScatterCardConfig,
        {
          type: RunsCompareChartType.PARALLEL,
          selectedParams: [],
          selectedMetrics: [],
        } as RunsCompareParallelCardConfig,
        { type: RunsCompareChartType.BAR },
      ],
      runs as CompareChartRunData[],
    );

    // Expect two bar charts in the set
    expect(wrapper.find(RunsCompareBarChartCard).length).toBe(2);

    // Expect one line chart in the set
    expect(wrapper.find(RunsCompareLineChartCard).length).toBe(1);

    // Expect one scatter chart in the set
    expect(wrapper.find(RunsCompareScatterChartCard).length).toBe(1);

    // Expect one contour chart in the set
    expect(wrapper.find(RunsCompareContourChartCard).length).toBe(1);

    // Expect one parallel coordinates chart in the set
    expect(wrapper.find(RunsCompareParallelChartCard).length).toBe(1);

    const allChartComponents = [
      wrapper.find(RunsCompareBarChartCard),
      wrapper.find(RunsCompareLineChartCard),
      wrapper.find(RunsCompareScatterChartCard),
      wrapper.find(RunsCompareContourChartCard),
      wrapper.find(RunsCompareParallelChartCard),
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
});

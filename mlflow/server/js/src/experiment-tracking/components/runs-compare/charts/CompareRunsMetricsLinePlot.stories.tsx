import { useCallback, useState } from 'react';
import { RunInfoEntity } from '../../../types';
import {
  chartColors,
  ChartStoryWrapper,
  getRandomRunName,
  stableNormalRandom,
} from './CompareRunsCharts.stories-common';
import {
  CompareRunsMetricsLinePlot,
  CompareRunsMetricsLinePlotProps,
} from './CompareRunsMetricsLinePlot';

export default {
  title: 'Compare runs charts/Metrics/Line plot',
  component: CompareRunsMetricsLinePlot,
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const createMockMetricsData = (
  numRuns: number,
  numValues: number,
  negative = false,
): CompareRunsMetricsLinePlotProps['runsData'] => {
  const random = stableNormalRandom(100);
  const refDate = new Date('2022-01-01T15:00:00');
  return new Array(numRuns).fill(0).map((_, index) => {
    const metricsHistory = new Array(numValues).fill(0).map((__, stepindex) => {
      let value = 500 * random() - 250;
      if (!negative) {
        value = Math.max(Math.abs(value), 100);
      }
      const timestamp = new Date(refDate.valueOf());
      timestamp.setSeconds(timestamp.getSeconds() + stepindex ** 2);

      return {
        step: stepindex + 1,
        timestamp,
        value,
      } as any;
    });

    const runName = getRandomRunName(random);

    return {
      runInfo: {
        run_uuid: `id-for-run-${runName}`,
        run_name: runName,
      } as RunInfoEntity,
      metricsHistory: { metric1: metricsHistory },
      color: chartColors[index % chartColors.length],
      metrics: {},
      params: {},
    };
  });
};

const DATA = createMockMetricsData(10, 10);
const NEGATIVE_DATA = createMockMetricsData(10, 10, true);

const MetricsRunWrapper = ({
  runsData,
  disableLog = false,
  xAxisKey,
  width,
  height,
}: Pick<CompareRunsMetricsLinePlotProps, 'runsData' | 'xAxisKey' | 'width' | 'height'> & {
  disableLog?: boolean;
}) => {
  const [log, setLog] = useState(false);
  const [polyLine, setPolyLine] = useState(false);
  const [hoveredRun, setHoveredRun] = useState('');

  const clear = useCallback(() => setHoveredRun(''), []);

  return (
    <ChartStoryWrapper
      title='Line chart'
      controls={
        <>
          {!disableLog && (
            <>
              Log scale:{' '}
              <input
                type='checkbox'
                checked={log}
                onChange={({ target: { checked } }) => setLog(checked)}
              />
            </>
          )}
          Poly line:{' '}
          <input
            type='checkbox'
            checked={polyLine}
            onChange={({ target: { checked } }) => setPolyLine(checked)}
          />
          hovered run: {hoveredRun}
        </>
      }
    >
      <CompareRunsMetricsLinePlot
        metricKey='metric1'
        runsData={runsData}
        scaleType={log ? 'log' : 'linear'}
        onHover={setHoveredRun}
        onUnhover={clear}
        lineShape={polyLine ? 'linear' : 'spline'}
        xAxisKey={xAxisKey}
        width={width}
        height={height}
      />
    </ChartStoryWrapper>
  );
};

export const TwoRuns = () => <MetricsRunWrapper runsData={DATA.slice(0, 2)} />;
export const TenRuns = () => <MetricsRunWrapper runsData={DATA} />;
export const TenRunsStatic = () => <MetricsRunWrapper runsData={DATA} />;
export const TenRunsInTimeDomain = () => <MetricsRunWrapper runsData={DATA} xAxisKey='time' />;
export const TenRunsNegative = () => <MetricsRunWrapper runsData={NEGATIVE_DATA} disableLog />;

TwoRuns.storyName = '2 runs (auto-size)';
TenRuns.storyName = '10 runs (auto-size)';
TenRunsStatic.storyName = '10 runs (static size: 600x200)';
TenRunsInTimeDomain.storyName = '10 runs with time on X axis (auto-size)';
TenRunsNegative.storyName = '10 runs with negative values (auto-size)';

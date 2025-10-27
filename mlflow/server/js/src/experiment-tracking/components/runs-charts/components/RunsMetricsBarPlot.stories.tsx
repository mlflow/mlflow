import { useState } from 'react';
import type { RunInfoEntity } from '../../../types';
import { chartColors, ChartStoryWrapper, getRandomRunName, stableNormalRandom } from './RunsCharts.stories-common';
import type { RunsMetricsBarPlotProps } from './RunsMetricsBarPlot';
import { RunsMetricsBarPlot } from './RunsMetricsBarPlot';

export default {
  title: 'Runs charts/Metrics/Bar plot',
  component: RunsMetricsBarPlot,
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const createMockMetricsData = (numRuns: number, negative = false): RunsMetricsBarPlotProps['runsData'] => {
  const random = stableNormalRandom(100);
  return new Array(numRuns).fill(0).map((_, index) => {
    let value = 500 * random() - 250;
    if (!negative) {
      value = Math.abs(value);
    }
    const runName = getRandomRunName(random);
    return {
      uuid: `id-for-run-${runName}`,
      displayName: runName,
      runInfo: {
        runUuid: `id-for-run-${runName}`,
        runName: runName,
      } as RunInfoEntity,
      metrics: { metric1: { value } as any },
      color: chartColors[index % chartColors.length],
    };
  });
};

const DATA = createMockMetricsData(10);
const NEGATIVE_DATA = createMockMetricsData(10, true);

const MetricsRunWrapper = ({
  runsData,
  metricKey,
  width,
  height,
  displayRunNames,
}: Pick<RunsMetricsBarPlotProps, 'runsData' | 'metricKey' | 'width' | 'height' | 'displayRunNames'>) => {
  const [hoveredRun, setHoveredRun] = useState('');
  return (
    <ChartStoryWrapper title="Line chart" controls={<>Hovered run ID: {hoveredRun}</>}>
      <RunsMetricsBarPlot
        metricKey={metricKey}
        runsData={runsData}
        onHover={setHoveredRun}
        onUnhover={() => setHoveredRun('')}
        width={width}
        height={height}
        displayRunNames={displayRunNames}
      />
    </ChartStoryWrapper>
  );
};

export const TwoRuns = () => <MetricsRunWrapper metricKey="metric1" runsData={DATA.slice(0, 2)} />;

export const TenRuns = () => <MetricsRunWrapper metricKey="metric1" runsData={DATA} />;
export const TenRunsNamesHidden = () => (
  <MetricsRunWrapper metricKey="metric1" runsData={DATA} displayRunNames={false} />
);
export const TenRunsStatic = () => <MetricsRunWrapper metricKey="metric1" runsData={DATA} width={300} height={500} />;

export const TenRunsNegative = () => <MetricsRunWrapper metricKey="metric1" runsData={NEGATIVE_DATA} />;

TwoRuns.storyName = '2 runs (auto-size)';
TenRuns.storyName = '10 runs (auto-size)';
TenRunsNamesHidden.storyName = '10 runs (auto-size, run names hidden)';
TenRunsStatic.storyName = '10 runs (static size: 300x500)';
TenRunsNegative.storyName = '10 runs with negative values (auto-size)';

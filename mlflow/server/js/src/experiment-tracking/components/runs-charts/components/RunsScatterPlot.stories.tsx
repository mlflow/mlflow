import { useCallback, useMemo, useState } from 'react';
import type { RunInfoEntity } from '../../../types';
import {
  chartColors,
  ChartStoryWrapper,
  getRandomRunName,
  stableNormalRandom,
  useControls,
} from './RunsCharts.stories-common';
import type { RunsScatterPlotProps } from './RunsScatterPlot';
import { RunsScatterPlot } from './RunsScatterPlot';

export default {
  title: 'Runs charts/Scatter plot',
  component: RunsScatterPlot,
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const createMockContourData = (numRuns: number): RunsScatterPlotProps['runsData'] => {
  const random = stableNormalRandom(0);
  return new Array(numRuns).fill(0).map((_, index) => {
    const runName = getRandomRunName(random);
    return {
      uuid: `id-for-run-${runName}`,
      displayName: runName,
      runInfo: {
        runUuid: `id-for-run-${runName}`,
        runName: runName,
      } as RunInfoEntity,
      metrics: {
        metric1: { key: 'metric1', value: Math.abs(500 * random() - 250) },
        metric2: { key: 'metric2', value: Math.abs(500 * random() - 250) },
        metric3: { key: 'metric3', value: Math.abs(500 * random() - 250) },
      } as any,
      params: {
        param1: { key: 'param1', value: Math.abs(500 * random() - 250) },
        param2: { key: 'param2', value: Math.abs(500 * random() - 250) },
        param3: { key: 'param3', value: Math.abs(500 * random() - 250) },
      } as any,
      tags: {} as any,
      images: {} as any,
      color: chartColors[index % chartColors.length],
    };
  });
};

const ScatterPlotStoryWrapper = (props: any) => {
  const { axisProps, controls } = useControls(false);
  const [hoveredRun, setHoveredRun] = useState('');

  const clear = useCallback(() => setHoveredRun(''), []);

  return (
    <ChartStoryWrapper
      title={props.title}
      controls={
        <>
          {controls}
          Hovered run ID: {hoveredRun}
        </>
      }
    >
      <RunsScatterPlot {...axisProps} onHover={setHoveredRun} onUnhover={clear} {...props} />
    </ChartStoryWrapper>
  );
};

export const FiftyRuns = () => <ScatterPlotStoryWrapper runsData={useMemo(() => createMockContourData(50), [])} />;

export const TwoHundredFiftyRuns = () => (
  <ScatterPlotStoryWrapper runsData={useMemo(() => createMockContourData(250), [])} />
);

export const TwoHundredFiftyRunsStatic = () => (
  <ScatterPlotStoryWrapper runsData={useMemo(() => createMockContourData(250), [])} width={500} height={250} />
);

export const ThousandRuns = () => <ScatterPlotStoryWrapper runsData={useMemo(() => createMockContourData(1000), [])} />;

FiftyRuns.storyName = '50 runs (auto-size)';
TwoHundredFiftyRuns.storyName = '250 runs (auto-size)';
TwoHundredFiftyRunsStatic.storyName = '250 runs (static size: 500x250)';
ThousandRuns.storyName = '1000 runs (auto-size)';

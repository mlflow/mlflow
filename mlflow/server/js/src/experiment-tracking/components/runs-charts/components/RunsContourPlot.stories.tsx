import { useCallback, useMemo, useState } from 'react';
import type { RunInfoEntity } from '../../../types';
import {
  chartColors,
  ChartStoryWrapper,
  getRandomRunName,
  stableNormalRandom,
  useControls,
} from './RunsCharts.stories-common';
import type { RunsContourPlotProps } from './RunsContourPlot';
import { RunsContourPlot } from './RunsContourPlot';

export default {
  title: 'Runs charts/Contour plot',
  component: RunsContourPlot,
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const createMockContourData = (numRuns: number): RunsContourPlotProps['runsData'] => {
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

const ContourPlotStoryWrapper = (props: any) => {
  const [reverse, setReverse] = useState(false);
  const { axisProps, controls } = useControls(true);
  const [hoveredRun, setHoveredRun] = useState('');

  const clear = useCallback(() => setHoveredRun(''), []);

  return (
    <ChartStoryWrapper
      title={props.title}
      controls={
        <span>
          {controls} Reverse scale:{' '}
          <input type="checkbox" checked={reverse} onChange={({ target }) => setReverse(target.checked)} /> Hovered run
          ID: {hoveredRun}
        </span>
      }
    >
      <RunsContourPlot reverseScale={reverse} {...axisProps} onHover={setHoveredRun} onUnhover={clear} {...props} />
    </ChartStoryWrapper>
  );
};
export const TenRuns = () => <ContourPlotStoryWrapper runsData={useMemo(() => createMockContourData(10), [])} />;
export const TenRunsStatic = () => (
  <ContourPlotStoryWrapper runsData={useMemo(() => createMockContourData(10), [])} width={400} height={400} />
);
export const SeventyRuns = () => <ContourPlotStoryWrapper runsData={useMemo(() => createMockContourData(70), [])} />;
export const CustomScaleRuns = () => (
  <ContourPlotStoryWrapper
    runsData={useMemo(() => createMockContourData(10), [])}
    colorScale={useMemo(
      () => [
        [0, 'rgb(0,0,224)'],
        [0.25, 'rgb(0,128,192)'],
        [0.5, 'rgb(255,0,0)'],
        [0.75, 'rgb(192,168,0)'],
        [1, 'rgb(192,168,0)'],
      ],
      [],
    )}
  />
);

TenRuns.storyName = '10 runs (auto-size)';
TenRunsStatic.storyName = '10 runs (static size: 400x400)';
SeventyRuns.storyName = '70 runs (auto-size)';
CustomScaleRuns.storyName = 'Custom color scale (auto-size)';

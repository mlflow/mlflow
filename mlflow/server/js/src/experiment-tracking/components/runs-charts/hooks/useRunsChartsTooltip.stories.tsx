import { Button, CloseIcon } from '@databricks/design-system';
import React, { PropsWithChildren, ReactNode, useState } from 'react';
import { IntlProvider } from 'react-intl';
import type { RunInfoEntity } from '../../../types';
import { chartColors, getRandomRunName, stableNormalRandom } from '../components/RunsCharts.stories-common';
import { RunsContourPlot } from '../components/RunsContourPlot';
import { RunsMetricsBarPlot } from '../components/RunsMetricsBarPlot';
import { RunsMetricsLinePlot } from '../components/RunsMetricsLinePlot';
import { RunsScatterPlot } from '../components/RunsScatterPlot';
import type { RunsChartsTooltipBodyProps } from './useRunsChartsTooltip';
import { RunsChartsTooltipWrapper, useRunsChartsTooltip } from './useRunsChartsTooltip';

export default {
  title: 'Runs charts/Context menu',
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const ContextMenuComponent = ({
  runUuid,
  contextData,
  hoverData,
  isHovering,
  closeContextMenu,
}: RunsChartsTooltipBodyProps<{ runs: ReturnType<typeof createMockData> }>) => {
  const run = contextData?.runs.find((x: any) => x.runInfo.runUuid === runUuid);

  if (!run) {
    return null;
  }

  return (
    <div>
      <div>name: {run.runInfo.runName}</div>
      <div>uuid: {runUuid}</div>
      <div>hovered menu id: {hoverData}</div>
      <div>mode: {isHovering ? 'hovering' : 'context menu'}</div>
      {!isHovering && (
        <div css={{ marginTop: 8 }}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_hooks_userunschartstooltip.stories.tsx_42"
            onClick={closeContextMenu}
            icon={<CloseIcon />}
          >
            Close
          </Button>
        </div>
      )}
    </div>
  );
};

/**
 * Some mock data (both line- and scatter-compatible)
 */
const createMockData = (numRuns: number, numValues: number, negative = false) => {
  const random = stableNormalRandom(100);
  const refDate = new Date('2022-01-01T15:00:00');
  return new Array(numRuns).fill(0).map((_, index) => {
    const metricsHistory = new Array(numValues).fill(0).map((__, stepindex) => {
      let value = 500 * random() - 250;
      if (!negative) {
        value = Math.abs(value);
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
        runUuid: `id-for-run-${runName}`,
        runName: runName,
      } as RunInfoEntity,
      metricsHistory: { metric1: metricsHistory },
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
      color: chartColors[index % chartColors.length],
    };
  });
};

/**
 * Just a HOC for story purposes for easy context wrap
 */
const withChartMenuContext =
  <
    T,
    P extends JSX.IntrinsicAttributes &
      JSX.LibraryManagedAttributes<React.ComponentType<React.PropsWithChildren<T>>, React.PropsWithChildren<T>>,
  >(
    Component: React.ComponentType<React.PropsWithChildren<T>>,
  ) =>
  (props: P) => {
    const [data, setData] = useState(() => ({ runs: createMockData(15, 10) }));

    return (
      <RunsChartsTooltipWrapper component={ContextMenuComponent} contextData={data}>
        <IntlProvider locale="en">
          <div css={{ padding: 16 }}>
            <div css={{ marginBottom: 16, height: 200, overflowY: 'scroll' }}>
              <ul>
                {data.runs
                  .slice()
                  .reverse()
                  .map((run) => (
                    <li key={run.runInfo.runUuid} style={{ fontWeight: 'bold', color: run.color }}>
                      {run.runInfo.runName}
                    </li>
                  ))}
              </ul>
            </div>
            <div css={{ margin: '8px 0', display: 'flex', gap: 8 }}>
              <button onClick={() => setData({ runs: createMockData(15, 10) })}>Dataset: default</button>
              <button onClick={() => setData({ runs: createMockData(15, 10).reverse() })}>Dataset: reverse</button>
              <button onClick={() => setData({ runs: createMockData(5, 10) })}>Dataset: 5 runs</button>
              <button onClick={() => setData({ runs: createMockData(30, 10) })}>Dataset: 30 runs</button>
            </div>
            <Component {...props} data={data} />
          </div>
        </IntlProvider>
      </RunsChartsTooltipWrapper>
    );
  };

export const ChartContextMenuStory = withChartMenuContext(({ data }: any) => {
  const { setTooltip, resetTooltip } = useRunsChartsTooltip('dummy-chart-id');

  return (
    <div css={styles.chartsGrid}>
      <div css={styles.chartWrapper}>
        <RunsMetricsBarPlot
          metricKey="metric1"
          runsData={data.runs}
          useDefaultHoverBox={false}
          displayRunNames={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          height={400}
          margin={{
            t: 0,
            b: 48,
            r: 0,
            l: 8,
            pad: 0,
          }}
        />
      </div>
      <div css={styles.chartWrapper}>
        <RunsScatterPlot
          xAxis={{ key: 'param1', type: 'PARAM' }}
          yAxis={{ key: 'param2', type: 'PARAM' }}
          runsData={data.runs}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          height={400}
        />
      </div>
      <div css={styles.chartWrapper}>
        <RunsMetricsLinePlot
          metricKey="metric1"
          selectedXAxisMetricKey=""
          runsData={data.runs}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          height={400}
        />
      </div>
      <div css={styles.chartWrapper}>
        <RunsContourPlot
          xAxis={{ key: 'param3', type: 'PARAM' }}
          yAxis={{ key: 'param2', type: 'PARAM' }}
          zAxis={{ key: 'param1', type: 'PARAM' }}
          runsData={data.runs}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          height={400}
        />
      </div>
    </div>
  );
});

const styles = {
  chartsGrid: { overflow: 'hidden', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 },
  chartWrapper: {
    overflow: 'hidden',
    border: `1px solid #ccc`,
    padding: 16,
    backgroundColor: 'white',
  },
};

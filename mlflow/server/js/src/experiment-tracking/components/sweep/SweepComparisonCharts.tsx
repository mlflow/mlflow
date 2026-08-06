import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import { LazyPlot } from '../LazyPlot';
import type { RunEntity } from '../../types';
import { useDynamicPlotSize } from '../runs-charts/components/RunsCharts.common';
import { useGetExperimentRunColor } from '../experiment-page/hooks/useExperimentRunColor';
import { buildSweepChartData, buildTradeoffPoints, type SweepChartPoint, type SweepRunSeries } from './sweepChartData';

const CHART_HEIGHT = 320;
const CHART_PREFERRED_WIDTH = 420;

const PLOT_CONFIG = { responsive: true, displaylogo: false, displayModeBar: false } as const;

/** Legend above the plot, so adding runs does not squeeze the plotting area horizontally. */
const LEGEND_LAYOUT = { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } } as const;

/**
 * Categorical hues for the per-scorer lines on the tradeoff charts, assigned in fixed order.
 *
 * Validated for colour-vision deficiency separation against the chart surface. Scorers past the
 * eighth reuse hues, which is preferable to generating unvalidated ones; the legend and hover label
 * still identify every line.
 */
const SCORER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#8f2c78'] as const;

/**
 * Wraps a plot in a size-observed container.
 *
 * Plotly fixes a plot's width when it first draws, so a chart laid out before its flex container
 * has settled keeps that stale width and paints content outside the plot area until the view is
 * reset (a double-click). Feeding the observed width back into the layout, together with
 * `autosize`, keeps each plot matched to its card.
 */
const SizedPlot = ({ data, layout }: { data: unknown[]; layout: Record<string, unknown> }) => {
  const { setContainerDiv, layoutWidth } = useDynamicPlotSize();

  return (
    <div ref={setContainerDiv} css={{ width: '100%', height: CHART_HEIGHT }}>
      <LazyPlot
        data={data}
        layout={{ ...layout, autosize: true, height: CHART_HEIGHT, width: layoutWidth }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

/**
 * Charts comparing the configurations of `mlflow.genai.evaluate_sweep` runs.
 *
 * Three views, in the order the comparison is usually made:
 *
 * 1. Per-scorer quality intervals — each config's mean with its 95% confidence interval as an error
 *    bar. Overlapping bars are the visual form of "these two are not distinguishable", which a
 *    table of numbers can only state in prose.
 * 2. Cost and latency per config, since both are per-config rather than per-scorer.
 * 3. Score against cost and against latency, one chart each, with a coloured line per scorer
 *    joining the configs. A line's slope reads as quality gained per extra dollar or millisecond,
 *    and comparing lines shows where scorers disagree about which config is worth it.
 *
 * These are purpose-built rather than added to the generic runs charts because a sweep's configs
 * live inside one run's metrics, while the runs charts plot one point per run.
 */
const ChartCard = ({ title, children }: { title: React.ReactNode; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        // Prefer two cards per row, but `minWidth: 0` lets a card shrink below that when the panel
        // is narrow (the runs page shows these beside the runs table) instead of overflowing it.
        flex: `1 1 ${CHART_PREFERRED_WIDTH}px`,
        minWidth: 0,
        padding: theme.spacing.sm,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
      }}
    >
      <Typography.Title level={4} withoutMargins css={{ marginBottom: theme.spacing.xs }}>
        {title}
      </Typography.Title>
      {children}
    </div>
  );
};

/**
 * Mean per config for one scorer, with the 95% confidence interval as an asymmetric error bar.
 *
 * Configs share the x-axis and each sweep run is its own series, so comparing the same config
 * across runs is a vertical read at one tick rather than hunting through interleaved labels.
 */
const ScorerIntervalChart = ({
  runSeries,
  configNames,
  scorer,
  showLegend,
  getRunColor,
}: {
  runSeries: SweepRunSeries[];
  configNames: string[];
  scorer: string;
  showLegend: boolean;
  getRunColor: (runUuid: string) => string;
}) => {
  const intl = useIntl();

  const plotData = useMemo(
    () =>
      runSeries.flatMap((series) => {
        const withScorer = series.points.filter((point) => point.scorers[scorer]?.mean !== undefined);
        if (withScorer.length === 0) {
          return [];
        }
        const means = withScorer.map((point) => point.scorers[scorer].mean as number);
        const color = getRunColor(series.runUuid);

        return [
          {
            x: withScorer.map((point) => point.config),
            y: means,
            name: series.runName,
            type: 'scatter' as const,
            mode: 'markers' as const,
            marker: { size: 10, color },
            error_y: {
              type: 'data' as const,
              symmetric: false,
              // Plotly wants distances from the point, not absolute bounds.
              array: withScorer.map((point, i) => (point.scorers[scorer].ciHigh ?? means[i]) - means[i]),
              arrayminus: withScorer.map((point, i) => means[i] - (point.scorers[scorer].ciLow ?? means[i])),
              color,
              thickness: 1.5,
              width: 6,
            },
            hovertemplate: `%{x}<br>%{y:.3f}<extra>${series.runName}</extra>`,
          },
        ];
      }),
    [runSeries, scorer, getRunColor],
  );

  return (
    <SizedPlot
      data={plotData}
      layout={{
        margin: { t: 8, r: 16, b: 80, l: 56 },
        showlegend: showLegend,
        legend: LEGEND_LAYOUT,
        // Pin the categories so every chart orders configs identically, and so a config missing
        // from one run still holds its slot instead of shifting the other series along.
        xaxis: {
          automargin: true,
          tickangle: -30,
          type: 'category',
          categoryorder: 'array',
          categoryarray: configNames,
        },
        yaxis: {
          automargin: true,
          title: intl.formatMessage({
            defaultMessage: 'Score',
            description: 'Evaluation runs page > sweep charts > scorer interval chart > y axis title',
          }),
        },
      }}
    />
  );
};

/** Grouped bar chart of a single per-config measure (cost or one latency percentile), one group per run. */
const PerConfigBarChart = ({
  runSeries,
  configNames,
  valueOf,
  axisTitle,
  hoverFormat,
  showLegend,
  getRunColor,
}: {
  runSeries: SweepRunSeries[];
  configNames: string[];
  valueOf: (point: SweepChartPoint) => number | undefined;
  axisTitle: string;
  hoverFormat: string;
  showLegend: boolean;
  getRunColor: (runUuid: string) => string;
}) => {
  const plotData = useMemo(
    () =>
      runSeries.flatMap((series) => {
        const withValue = series.points.filter((point) => valueOf(point) !== undefined);
        if (withValue.length === 0) {
          return [];
        }
        return [
          {
            x: withValue.map((point) => point.config),
            y: withValue.map((point) => valueOf(point) as number),
            name: series.runName,
            type: 'bar' as const,
            marker: { color: getRunColor(series.runUuid) },
            hovertemplate: `%{x}<br>${hoverFormat}<extra>${series.runName}</extra>`,
          },
        ];
      }),
    [runSeries, valueOf, hoverFormat, getRunColor],
  );

  return (
    <SizedPlot
      data={plotData}
      layout={{
        margin: { t: 8, r: 16, b: 80, l: 64 },
        showlegend: showLegend,
        legend: LEGEND_LAYOUT,
        barmode: 'group',
        xaxis: {
          automargin: true,
          tickangle: -30,
          type: 'category',
          categoryorder: 'array',
          categoryarray: configNames,
        },
        yaxis: { automargin: true, title: axisTitle, rangemode: 'tozero' },
      }}
    />
  );
};

/**
 * Score against cost or latency, with one coloured line per scorer.
 *
 * Each point is a (config, scorer) pair; the line joins the configs for one scorer, so its slope
 * reads directly as "how much quality do I gain per extra dollar / millisecond". Comparing lines
 * shows where scorers disagree — a config can be the best choice on one and dominated on another.
 *
 * Configs are ordered along the x-axis by cost/latency, so the line always runs cheap-to-expensive
 * (or fast-to-slow) and never doubles back on itself.
 */
const TradeoffChart = ({
  points,
  scorerNames,
  costOf,
  xAxisTitle,
  xHoverFormat,
}: {
  points: SweepChartPoint[];
  scorerNames: string[];
  costOf: (point: SweepChartPoint) => number | undefined;
  xAxisTitle: string;
  xHoverFormat: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { plotData, ranges } = useMemo(() => {
    const allCosts: number[] = [];
    const allScores: number[] = [];

    const data = scorerNames.flatMap((scorer, index) => {
      const tradeoffs = buildTradeoffPoints(points, scorer, costOf).sort((a, b) => a.cost - b.cost);
      if (tradeoffs.length === 0) {
        return [];
      }
      const color = SCORER_COLORS[index % SCORER_COLORS.length];
      allCosts.push(...tradeoffs.map((point) => point.cost));
      allScores.push(...tradeoffs.map((point) => point.ciHigh ?? point.score));
      allScores.push(...tradeoffs.map((point) => point.ciLow ?? point.score));

      return [
        {
          x: tradeoffs.map((point) => point.cost),
          y: tradeoffs.map((point) => point.score),
          text: tradeoffs.map((point) => point.label),
          name: scorer,
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          line: { color, width: 2 },
          marker: {
            size: 10,
            color,
            // A ring in the surface colour separates markers that overlap between scorers.
            line: { color: theme.colors.backgroundPrimary, width: 2 },
          },
          error_y: {
            type: 'data' as const,
            symmetric: false,
            array: tradeoffs.map((point) => (point.ciHigh ?? point.score) - point.score),
            arrayminus: tradeoffs.map((point) => point.score - (point.ciLow ?? point.score)),
            color,
            thickness: 1,
            width: 4,
            opacity: 0.5,
          },
          hovertemplate: `%{text}<br>${xHoverFormat}<br>%{y:.3f}<extra>${scorer}</extra>`,
        },
      ];
    });

    // Plotly excludes the error bars and any overhanging marker ring from its autoscale, so pad.
    const pad = (values: number[], fraction: number) => {
      if (values.length === 0) {
        return { min: 0, max: 1 };
      }
      const min = Math.min(...values);
      const max = Math.max(...values);
      const spread = max - min || Math.abs(max) || 1;
      return { min: min - spread * fraction, max: max + spread * fraction };
    };

    return {
      plotData: data,
      ranges: { cost: pad(allCosts, 0.1), score: pad(allScores, 0.1) },
    };
  }, [points, scorerNames, costOf, xHoverFormat, theme.colors.backgroundPrimary]);

  return (
    <SizedPlot
      data={plotData}
      layout={{
        margin: { t: 8, r: 24, b: 56, l: 64 },
        hovermode: 'closest',
        showlegend: true,
        legend: LEGEND_LAYOUT,
        // Cost and latency are never negative, so do not pad below zero.
        xaxis: { automargin: true, title: xAxisTitle, range: [Math.max(0, ranges.cost.min), ranges.cost.max] },
        yaxis: {
          automargin: true,
          range: [ranges.score.min, ranges.score.max],
          title: intl.formatMessage({
            defaultMessage: 'Score',
            description: 'Evaluation runs page > sweep charts > tradeoff chart > y axis title',
          }),
        },
      }}
    />
  );
};

const costOfPoint = (point: SweepChartPoint) => point.costPerRequestUsd;
const latencyOfPoint = (point: SweepChartPoint) => point.latency?.p50;

export const SweepComparisonCharts = ({ runs = [] }: { runs?: RunEntity[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { points, runSeries, configNames, scorerNames } = useMemo(() => buildSweepChartData(runs), [runs]);
  // Reuse the run colours from the runs table, so a run reads the same in both.
  const getRunColor = useGetExperimentRunColor();

  const anyCost = points.some((point) => point.costPerRequestUsd !== undefined);
  const anyLatency = points.some((point) => point.latency?.p50 !== undefined);
  // With one sweep the legend would just repeat the run name already shown in the table.
  const showLegend = runSeries.length > 1;

  if (points.length === 0) {
    return null;
  }

  const axisTitles = {
    cost: intl.formatMessage({
      defaultMessage: 'Cost per request (USD)',
      description: 'Evaluation runs page > sweep charts > cost axis title',
    }),
    latency: intl.formatMessage({
      defaultMessage: 'Latency p50 (ms)',
      description: 'Evaluation runs page > sweep charts > latency axis title',
    }),
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <Typography.Title level={3} withoutMargins>
          <FormattedMessage
            defaultMessage="Evaluation sweep comparison"
            description="Evaluation runs page > sweep charts > section title"
          />
        </Typography.Title>
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Each point is one swept configuration. Error bars are 95% confidence intervals — configurations whose bars overlap are not distinguishable at that confidence."
            description="Evaluation runs page > sweep charts > section description"
          />
        </Typography.Hint>
      </div>

      <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.md }}>
        {scorerNames.map((scorer) => (
          <ChartCard
            key={scorer}
            title={intl.formatMessage(
              {
                defaultMessage: '{scorer} by configuration',
                description: 'Evaluation runs page > sweep charts > per-scorer chart title',
              },
              { scorer },
            )}
          >
            <ScorerIntervalChart
              runSeries={runSeries}
              configNames={configNames}
              scorer={scorer}
              showLegend={showLegend}
              getRunColor={getRunColor}
            />
          </ChartCard>
        ))}

        {anyCost && (
          <ChartCard
            title={
              <FormattedMessage
                defaultMessage="Cost per request"
                description="Evaluation runs page > sweep charts > cost chart title"
              />
            }
          >
            <PerConfigBarChart
              runSeries={runSeries}
              configNames={configNames}
              valueOf={costOfPoint}
              axisTitle={axisTitles.cost}
              hoverFormat="$%{y:.4f}"
              showLegend={showLegend}
              getRunColor={getRunColor}
            />
          </ChartCard>
        )}

        {anyLatency && (
          <ChartCard
            title={
              <FormattedMessage
                defaultMessage="Latency p50"
                description="Evaluation runs page > sweep charts > latency chart title"
              />
            }
          >
            <PerConfigBarChart
              runSeries={runSeries}
              configNames={configNames}
              valueOf={latencyOfPoint}
              axisTitle={axisTitles.latency}
              hoverFormat="%{y:.0f} ms"
              showLegend={showLegend}
              getRunColor={getRunColor}
            />
          </ChartCard>
        )}

        {anyCost && (
          <ChartCard
            title={
              <FormattedMessage
                defaultMessage="Score vs cost"
                description="Evaluation runs page > sweep charts > score vs cost chart title"
              />
            }
          >
            <TradeoffChart
              points={points}
              scorerNames={scorerNames}
              costOf={costOfPoint}
              xAxisTitle={axisTitles.cost}
              xHoverFormat="$%{x:.4f}"
            />
          </ChartCard>
        )}

        {anyLatency && (
          <ChartCard
            title={
              <FormattedMessage
                defaultMessage="Score vs latency"
                description="Evaluation runs page > sweep charts > score vs latency chart title"
              />
            }
          >
            <TradeoffChart
              points={points}
              scorerNames={scorerNames}
              costOf={latencyOfPoint}
              xAxisTitle={axisTitles.latency}
              xHoverFormat="%{x:.0f} ms"
            />
          </ChartCard>
        )}
      </div>

      {!anyCost && !anyLatency && (
        <Empty
          description={
            <FormattedMessage
              defaultMessage="This sweep logged no latency or cost data, so only quality is shown. Cost requires a provider that reports token usage."
              description="Evaluation runs page > sweep charts > missing latency and cost notice"
            />
          }
        />
      )}
    </div>
  );
};

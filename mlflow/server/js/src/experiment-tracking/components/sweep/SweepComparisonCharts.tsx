import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import { LazyPlot } from '../LazyPlot';
import type { RunEntity } from '../../types';
import { buildSweepChartData, buildTradeoffPoints, type SweepChartPoint } from './sweepChartData';

const CHART_HEIGHT = 320;
const CHART_MIN_WIDTH = 420;

const PLOT_CONFIG = { responsive: true, displaylogo: false, displayModeBar: false } as const;

/**
 * Charts comparing the configurations of `mlflow.genai.evaluate_sweep` runs.
 *
 * Three views, in the order the comparison is usually made:
 *
 * 1. Per-scorer quality intervals — each config's mean with its 95% confidence interval as an error
 *    bar. Overlapping bars are the visual form of "these two are not distinguishable", which a
 *    table of numbers can only state in prose.
 * 2. Cost and latency per config, since both are per-config rather than per-scorer.
 * 3. Score-vs-cost and score-vs-latency tradeoffs, with the Pareto frontier joined by a line, to
 *    show how much quality a cheaper or faster config gives up.
 *
 * These are purpose-built rather than added to the generic runs charts because a sweep's configs
 * live inside one run's metrics, while the runs charts plot one point per run.
 */
const ChartCard = ({ title, children }: { title: React.ReactNode; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        flex: `1 1 ${CHART_MIN_WIDTH}px`,
        minWidth: CHART_MIN_WIDTH,
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

/** Mean per config for one scorer, with the 95% confidence interval as an asymmetric error bar. */
const ScorerIntervalChart = ({ points, scorer }: { points: SweepChartPoint[]; scorer: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const plotData = useMemo(() => {
    const withScorer = points.filter((point) => point.scorers[scorer]?.mean !== undefined);
    const means = withScorer.map((point) => point.scorers[scorer].mean as number);

    return [
      {
        x: withScorer.map((point) => point.label),
        y: means,
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: { size: 10, color: theme.colors.primary },
        error_y: {
          type: 'data' as const,
          symmetric: false,
          // Plotly wants distances from the point, not absolute bounds.
          array: withScorer.map((point, i) => (point.scorers[scorer].ciHigh ?? means[i]) - means[i]),
          arrayminus: withScorer.map((point, i) => means[i] - (point.scorers[scorer].ciLow ?? means[i])),
          color: theme.colors.primary,
          thickness: 1.5,
          width: 6,
        },
        hovertemplate: '%{x}<br>%{y:.3f}<extra></extra>',
      },
    ];
  }, [points, scorer, theme.colors.primary]);

  return (
    <LazyPlot
      data={plotData}
      layout={{
        height: CHART_HEIGHT,
        margin: { t: 8, r: 8, b: 80, l: 48 },
        showlegend: false,
        xaxis: { automargin: true, tickangle: -30 },
        yaxis: {
          title: intl.formatMessage({
            defaultMessage: 'Score',
            description: 'Evaluation runs page > sweep charts > scorer interval chart > y axis title',
          }),
        },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
};

/** Bar chart of a single per-config measure (cost or one latency percentile). */
const PerConfigBarChart = ({
  points,
  valueOf,
  axisTitle,
  hoverFormat,
}: {
  points: SweepChartPoint[];
  valueOf: (point: SweepChartPoint) => number | undefined;
  axisTitle: string;
  hoverFormat: string;
}) => {
  const { theme } = useDesignSystemTheme();

  const plotData = useMemo(() => {
    const withValue = points.filter((point) => valueOf(point) !== undefined);
    return [
      {
        x: withValue.map((point) => point.label),
        y: withValue.map((point) => valueOf(point) as number),
        type: 'bar' as const,
        marker: { color: theme.colors.primary },
        hovertemplate: `%{x}<br>${hoverFormat}<extra></extra>`,
      },
    ];
  }, [points, valueOf, hoverFormat, theme.colors.primary]);

  return (
    <LazyPlot
      data={plotData}
      layout={{
        height: CHART_HEIGHT,
        margin: { t: 8, r: 8, b: 80, l: 56 },
        showlegend: false,
        xaxis: { automargin: true, tickangle: -30 },
        yaxis: { title: axisTitle, rangemode: 'tozero' },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
};

/**
 * Score against cost or latency, one point per config, with the Pareto frontier joined.
 *
 * Points below the frontier are dominated — something else is both cheaper (or faster) and at least
 * as accurate — so the frontier is what a reader should choose between.
 */
const TradeoffChart = ({
  points,
  scorer,
  costOf,
  xAxisTitle,
  xHoverFormat,
}: {
  points: SweepChartPoint[];
  scorer: string;
  costOf: (point: SweepChartPoint) => number | undefined;
  xAxisTitle: string;
  xHoverFormat: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const plotData = useMemo(() => {
    const tradeoffs = buildTradeoffPoints(points, scorer, costOf);
    const frontier = tradeoffs.filter((point) => point.isOnFrontier).sort((a, b) => a.cost - b.cost);

    return [
      {
        x: frontier.map((point) => point.cost),
        y: frontier.map((point) => point.score),
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: theme.colors.border, dash: 'dot' as const, width: 1.5 },
        hoverinfo: 'skip' as const,
        showlegend: false,
      },
      {
        x: tradeoffs.map((point) => point.cost),
        y: tradeoffs.map((point) => point.score),
        text: tradeoffs.map((point) => point.label),
        type: 'scatter' as const,
        mode: 'markers+text' as const,
        textposition: 'top center' as const,
        textfont: { size: 10, color: theme.colors.textSecondary },
        marker: {
          size: 12,
          // Frontier configs are filled; dominated ones are hollow, so the defensible choices read
          // at a glance without needing the legend.
          color: tradeoffs.map((point) => (point.isOnFrontier ? theme.colors.primary : 'transparent')),
          line: { color: theme.colors.primary, width: 2 },
        },
        error_y: {
          type: 'data' as const,
          symmetric: false,
          array: tradeoffs.map((point) => (point.ciHigh ?? point.score) - point.score),
          arrayminus: tradeoffs.map((point) => point.score - (point.ciLow ?? point.score)),
          color: theme.colors.border,
          thickness: 1,
          width: 4,
        },
        hovertemplate: `%{text}<br>${xHoverFormat}<br>%{y:.3f}<extra></extra>`,
        showlegend: false,
      },
    ];
  }, [points, scorer, costOf, xHoverFormat, theme.colors]);

  return (
    <LazyPlot
      data={plotData}
      layout={{
        height: CHART_HEIGHT,
        margin: { t: 16, r: 16, b: 48, l: 56 },
        hovermode: 'closest',
        xaxis: { title: xAxisTitle, rangemode: 'tozero' },
        yaxis: {
          title: intl.formatMessage({
            defaultMessage: 'Score',
            description: 'Evaluation runs page > sweep charts > tradeoff chart > y axis title',
          }),
        },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
};

const costOfPoint = (point: SweepChartPoint) => point.costPerRequestUsd;
const latencyOfPoint = (point: SweepChartPoint) => point.latency?.p50;

export const SweepComparisonCharts = ({ runs = [] }: { runs?: RunEntity[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { points, scorerNames } = useMemo(() => buildSweepChartData(runs), [runs]);

  const anyCost = points.some((point) => point.costPerRequestUsd !== undefined);
  const anyLatency = points.some((point) => point.latency?.p50 !== undefined);

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
            <ScorerIntervalChart points={points} scorer={scorer} />
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
              points={points}
              valueOf={costOfPoint}
              axisTitle={axisTitles.cost}
              hoverFormat="$%{y:.4f}"
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
              points={points}
              valueOf={latencyOfPoint}
              axisTitle={axisTitles.latency}
              hoverFormat="%{y:.0f} ms"
            />
          </ChartCard>
        )}

        {scorerNames.flatMap((scorer) => [
          anyCost && (
            <ChartCard
              key={`${scorer}-cost`}
              title={intl.formatMessage(
                {
                  defaultMessage: '{scorer} vs cost',
                  description: 'Evaluation runs page > sweep charts > score vs cost chart title',
                },
                { scorer },
              )}
            >
              <TradeoffChart
                points={points}
                scorer={scorer}
                costOf={costOfPoint}
                xAxisTitle={axisTitles.cost}
                xHoverFormat="$%{x:.4f}"
              />
            </ChartCard>
          ),
          anyLatency && (
            <ChartCard
              key={`${scorer}-latency`}
              title={intl.formatMessage(
                {
                  defaultMessage: '{scorer} vs latency',
                  description: 'Evaluation runs page > sweep charts > score vs latency chart title',
                },
                { scorer },
              )}
            >
              <TradeoffChart
                points={points}
                scorer={scorer}
                costOf={latencyOfPoint}
                xAxisTitle={axisTitles.latency}
                xHoverFormat="%{x:.0f} ms"
              />
            </ChartCard>
          ),
        ])}
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

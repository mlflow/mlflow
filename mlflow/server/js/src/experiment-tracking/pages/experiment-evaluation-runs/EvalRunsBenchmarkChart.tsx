import {
  useDesignSystemTheme,
  ChartLineIcon,
  Empty,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
  Button,
  Typography,
} from '@databricks/design-system';
import React, { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useLocalStorage } from '@databricks/web-shared/hooks';

import type { RunEntity } from '../../types';
import { isLowerBetterMetric } from './EvalRunsBaseline.utils';
import { useChartColors, useLegendHighlight } from '../experiment-overview/utils/chartUtils';

/**
 * Props for the EvalRunsBenchmarkChart component
 */
export interface EvalRunsBenchmarkChartProps {
  runs: RunEntity[];
  /**
   * Passed in already resolved rather than looked up in `runs`: `runs` is the
   * filtered result, so a filtered-out baseline would leave the chart claiming no
   * baseline is set when one is.
   */
  baselineRun?: RunEntity;
  metricKeys: string[];
  /**
   * Runs ticked in the table. When non-empty the chart plots these instead of the
   * most recent runs, which turns it from a fixed summary into the answer to
   * "how do the ones I just picked compare" — the question the checkboxes imply.
   */
  selectedRunUuids?: string[];
}

/**
 * A single-hue ramp, light → dark, ordered oldest → newest. Because the default
 * runs are a sequence rather than unrelated categories, a categorical rainbow
 * would imply they are unrelated; darkening by recency makes "which is latest"
 * readable off the bars without consulting the legend.
 */
const RUN_COLORS = ['#BDD9EC', '#8BBDDE', '#5A9FCF', '#3182BD', '#1F5C8C'];

// Black reads as "the fixed reference" against the blue run ramp.
const BASELINE_COLOR = '#161616';

/**
 * How many runs the default (no selection) view plots.
 */
const DEFAULT_RUN_COUNT = 5;

/**
 * Ceiling on runs charted from an explicit selection. Selecting a whole page of
 * runs is one gesture, so the chart has to survive it: past roughly this many the
 * bars are narrower than the gaps between them and the legend is taller than the
 * plot. The overflow is named in a hint rather than silently dropped.
 */
const MAX_SELECTED_RUNS = 8;

/**
 * Plot height. Shared by the chart and its empty state so clearing the scorer
 * selection doesn't move the table below it.
 */
const CHART_HEIGHT = 340;

/**
 * Gap in pixels between bars inside one scorer cluster. Pinned to zero rather
 * than left at Recharts' default so the cluster reads as a single shape to
 * compare against its neighbours — and so the baseline's reference line can span
 * exactly `barWidth × seriesCount` with no gap arithmetic to get wrong.
 */
const BAR_GAP = 0;

/**
 * Helper to get the metric value for a run and metric key
 */
const getMetricValue = (run: RunEntity, metricKey: string): number | undefined => {
  const metric = run.data.metrics.find((m) => m.key === metricKey);
  return metric?.value;
};

/**
 * Extract dataset name from run
 */
const getDatasetName = (run: RunEntity): string | undefined => {
  return run.inputs?.datasetInputs?.[0]?.dataset?.name;
};

/**
 * One row per scorer: the scorer name plus each charted run's value for it, keyed
 * by run UUID. The run keys are dynamic, so they cannot be enumerated in the type.
 */
interface ChartRow {
  scorer: string;
  [runUuid: string]: string | number | undefined;
}

interface ChartSeries {
  runUuid: string;
  name: string;
  color: string;
  datasetName?: string;
}

/**
 * The baseline bar, plus a dashed rule across the whole scorer cluster at the
 * baseline's height.
 *
 * The rule is what lets you read "did this run beat the reference" off a single
 * cluster without tracing back to the black bar's top edge. It has to be drawn
 * here rather than as a `ReferenceLine`: a ReferenceLine spans the entire plot
 * width, so with several scorers on one axis it would draw one long rule at the
 * first scorer's value straight through clusters it says nothing about.
 *
 * Recharts 3 does not export the axis scales publicly, so the cluster's extent is
 * derived from the geometry Recharts already hands this shape: the baseline is
 * the leftmost series, so the cluster starts at its own `x` and — with
 * {@link BAR_GAP} pinned to zero — runs one `width` per bar to the right.
 *
 * The span counts the bars this scorer actually has, not every series on the
 * chart. Recharts reserves a slot per series whether or not that run recorded the
 * metric, so spanning the full cluster left the rule trailing across empty space
 * on a sparsely-recorded scorer — which reads as a rendering fault rather than as
 * missing data.
 */
const BaselineBarShape = ({
  x,
  y,
  width,
  height,
  fill,
  payload,
  runUuids = [],
}: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  fill?: string;
  payload?: ChartRow;
  /** Every series key on the chart, baseline included, for counting present bars. */
  runUuids?: string[];
}) => {
  if (x === undefined || y === undefined || width === undefined || height === undefined) {
    return null;
  }
  const presentBars = runUuids.filter((runUuid) => typeof payload?.[runUuid] === 'number').length || 1;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} />
      <line
        x1={x}
        x2={x + width * presentBars}
        y1={y}
        y2={y}
        stroke={BASELINE_COLOR}
        strokeWidth={1.5}
        strokeDasharray="4 3"
      />
    </g>
  );
};

/**
 * Collapsible benchmark chart comparing runs against a baseline. Each x-axis
 * group is a scorer; within a group there is one bar for the baseline (black) and
 * one per compared run.
 *
 * Which runs get compared depends on what the user has done: with rows ticked in
 * the table it plots those, otherwise the most recent few.
 */
export const EvalRunsBenchmarkChart = ({
  runs,
  baselineRun,
  metricKeys,
  selectedRunUuids,
}: EvalRunsBenchmarkChartProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { getChartColor } = useChartColors();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // State for collapsed/expanded
  const [isCollapsed, setIsCollapsed] = useLocalStorage({
    key: 'mlflow.evalRuns.benchmarkChart.collapsed',
    version: 1,
    initialValue: false,
  });

  // Default to bounded 0–1 scores only. One shared y-axis means an unbounded
  // metric like latency_p50_ms (~1200) flattens every score bar to nothing, so
  // those are available in the dropdown but never charted by default.
  const defaultMetricKeys = useMemo(() => {
    const scoreKeys = metricKeys.filter((key) => !isLowerBetterMetric(key));
    return (scoreKeys.length > 0 ? scoreKeys : metricKeys).slice(0, 4);
  }, [metricKeys]);

  // Null means "the user has not chosen yet", which is not the same as "the user
  // cleared the selection" — an empty array. The default has to be derived
  // rather than seeded into useState: on first render the runs have not loaded,
  // so metricKeys is empty, and a useState initializer would capture that empty
  // array once and never recover once the metrics arrived.
  const [userMetricKeys, setUserMetricKeys] = useState<string[] | null>(null);
  const selectedMetricKeys = userMetricKeys ?? defaultMetricKeys;

  // Search state for combobox
  const [search, setSearch] = useState('');

  const baselineRunUuid = baselineRun?.info.runUuid;

  // The baseline is never also a compared run: it is already drawn, and pairing
  // it with itself would add a bar whose delta is always zero.
  const selectedRuns = useMemo(() => {
    if (!selectedRunUuids?.length) {
      return [];
    }
    const picked = new Set(selectedRunUuids);
    return runs.filter((run) => picked.has(run.info.runUuid) && run.info.runUuid !== baselineRunUuid);
  }, [runs, selectedRunUuids, baselineRunUuid]);

  const isSelectionDriven = selectedRuns.length > 0;

  const chartedRuns = useMemo(() => {
    if (isSelectionDriven) {
      // Left in table order (newest first) rather than reversed: the selection is
      // a set the user assembled, not a timeline, so matching the order they see
      // above the chart is more useful than imposing a chronology on it.
      return selectedRuns.slice(0, MAX_SELECTED_RUNS);
    }
    const withoutBaseline = baselineRunUuid ? runs.filter((run) => run.info.runUuid !== baselineRunUuid) : runs;
    // Runs arrive sorted newest first; reverse so they read oldest → newest.
    return withoutBaseline.slice(0, DEFAULT_RUN_COUNT).reverse();
  }, [isSelectionDriven, selectedRuns, runs, baselineRunUuid]);

  const hiddenSelectedRunCount = isSelectionDriven ? selectedRuns.length - chartedRuns.length : 0;

  // A scorer is charted only if it forms an actual comparison: the baseline
  // recorded it AND at least one of the compared runs did. Either half missing
  // leaves a cluster of one bar trailed by a dashed line across empty space,
  // which looks like a rendering fault rather than absent data. The omission is
  // named in a hint below instead.
  const comparableMetricKeys = useMemo(
    () =>
      selectedMetricKeys.filter(
        (key) =>
          baselineRun !== undefined &&
          getMetricValue(baselineRun, key) !== undefined &&
          chartedRuns.some((run) => getMetricValue(run, key) !== undefined),
      ),
    [selectedMetricKeys, baselineRun, chartedRuns],
  );

  /**
   * Series are keyed by run UUID, not run name. Recharts reads a `dataKey` as a
   * path, so a run named `v1.2` would be looked up as `data['v1']['2']` and
   * silently plot nothing; UUIDs are also unique, which run names are not. The
   * name is carried separately for the legend and tooltip.
   */
  const series = useMemo<ChartSeries[]>(
    () =>
      chartedRuns.map((run, index) => ({
        runUuid: run.info.runUuid,
        name: run.info.runName ?? run.info.runUuid,
        // A recency ramp for the default view, where left-to-right *is* time, and
        // distinct hues for an explicit selection, where it is not — a sequence
        // shading would claim an order the user's picks do not have.
        color: isSelectionDriven ? getChartColor(index) : RUN_COLORS[index % RUN_COLORS.length],
        datasetName: getDatasetName(run),
      })),
    [chartedRuns, isSelectionDriven, getChartColor],
  );

  /**
   * Recharts is row-per-category, so this is one row per scorer carrying every
   * run's value for it — the transpose of Plotly's one-trace-per-run shape.
   */
  const chartData = useMemo(
    () =>
      comparableMetricKeys.map((metricKey) => {
        const row: ChartRow = { scorer: metricKey };
        if (baselineRun) {
          row[baselineRun.info.runUuid] = getMetricValue(baselineRun, metricKey);
        }
        chartedRuns.forEach((run) => {
          row[run.info.runUuid] = getMetricValue(run, metricKey);
        });
        return row;
      }),
    [comparableMetricKeys, baselineRun, chartedRuns],
  );

  // Bounded scores get a fixed 0–1 range so a given difference is the same height
  // on every render and across scorers; a metric outside that range (a rate, a
  // count) keeps Recharts' autoscaling.
  const isBoundedScoreAxis = useMemo(() => {
    const values = [baselineRun, ...chartedRuns]
      .filter((run): run is RunEntity => run !== undefined)
      .flatMap((run) => comparableMetricKeys.map((key) => getMetricValue(run, key)))
      .filter((value): value is number => value !== undefined);
    return values.length > 0 && values.every((value) => value >= 0 && value <= 1);
  }, [baselineRun, chartedRuns, comparableMetricKeys]);

  // Datasets are looked up by the series key the tooltip hands back. The compared
  // runs may span different datasets, in which case the bars are not strictly
  // comparable — so the dataset is named on every hover rather than left implied.
  const datasetByRunUuid = useMemo(() => {
    const map = new Map<string, string>();
    if (baselineRun) {
      map.set(baselineRun.info.runUuid, getDatasetName(baselineRun) ?? '—');
    }
    series.forEach((entry) => map.set(entry.runUuid, entry.datasetName ?? '—'));
    return map;
  }, [baselineRun, series]);

  const tooltipFormatter = useCallback(
    // Recharts types `value` and `name` as possibly-undefined and `dataKey` as
    // anything a dataKey may be, including a function — so both are narrowed here
    // rather than cast away.
    (value: number | undefined, name: string | undefined, item: { dataKey?: unknown }) => {
      const formatted = typeof value === 'number' ? value.toFixed(3) : '—';
      const dataset = typeof item?.dataKey === 'string' ? datasetByRunUuid.get(item.dataKey) : undefined;
      return [dataset ? `${formatted} · ${dataset}` : formatted, name ?? ''] as [string, string];
    },
    [datasetByRunUuid],
  );

  // Toggling resolves the default into an explicit selection, so the first click
  // edits what is on screen rather than starting from nothing.
  const handleScorerToggle = useCallback(
    (key: string) => {
      setUserMetricKeys((prev) => {
        const current = prev ?? defaultMetricKeys;
        return current.includes(key) ? current.filter((k) => k !== key) : [...current, key];
      });
    },
    [defaultMetricKeys],
  );

  const filteredMetricKeys = useMemo(() => {
    if (!search) return metricKeys;
    const trimmed = search.toLowerCase();
    return metricKeys.filter((key) => key.toLowerCase().includes(trimmed));
  }, [metricKeys, search]);

  const triggerLabel = useMemo(() => {
    return `${selectedMetricKeys.length} of ${metricKeys.length}`;
  }, [selectedMetricKeys.length, metricKeys.length]);

  const triggerValue = useMemo(
    () => (selectedMetricKeys.length > 0 ? [triggerLabel] : []),
    [selectedMetricKeys.length, triggerLabel],
  );

  const baselineLegendName = useMemo(
    () =>
      baselineRun
        ? `${baselineRun.info.runName} (${formatMessage({
            defaultMessage: 'baseline',
            description: 'Legend suffix marking which series in the benchmark chart is the baseline run',
          })})`
        : '',
    [baselineRun, formatMessage],
  );

  // Every series key on the chart, baseline first, matching the bar order. The
  // baseline's shape uses this to count how many bars its cluster really has.
  const allRunUuids = useMemo(
    () => (baselineRun ? [baselineRun.info.runUuid] : []).concat(series.map((entry) => entry.runUuid)),
    [baselineRun, series],
  );

  const scorerSelector = (
    <DialogCombobox
      componentId="mlflow.eval-runs.benchmark-chart.scorers-selector"
      label={formatMessage({
        defaultMessage: 'Scorers',
        description: 'Label for the scorer/metric selector in benchmark chart',
      })}
      multiSelect
      value={triggerValue}
    >
      {/* The label is inline on the trigger rather than a separate <label>
          element beside it, so the control reads as one compact chip in the
          header instead of two loose pieces. */}
      <DialogComboboxTrigger
        placeholder={formatMessage({
          defaultMessage: 'Select scorers',
          description: 'Placeholder for scorer selector',
        })}
        onClear={() => setUserMetricKeys([])}
      />
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
            {filteredMetricKeys.length === 0 ? (
              <DialogComboboxOptionListCheckboxItem value="" checked={false} onChange={() => {}} disabled>
                {search ? (
                  <FormattedMessage
                    defaultMessage="No matching scorers"
                    description="Message when no scorers match search"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="No scorers available"
                    description="Message when no scorers available"
                  />
                )}
              </DialogComboboxOptionListCheckboxItem>
            ) : (
              filteredMetricKeys.map((key) => (
                <DialogComboboxOptionListCheckboxItem
                  key={key}
                  value={key}
                  checked={selectedMetricKeys.includes(key)}
                  onChange={() => handleScorerToggle(key)}
                />
              ))
            )}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );

  // Render nothing if insufficient data
  if (runs.length < 2 || metricKeys.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        // A rule plus real breathing room above it. Measured, the chart title sat
        // 16px below the filter chips — closer to the toolbar than to its own
        // plot — so it read as one more row of controls rather than the start of
        // a new section. The border matches the page's existing section rules
        // (theme.colors.border, as used above the runs metadata).
        marginTop: theme.spacing.md,
        paddingTop: theme.spacing.md,
        borderTop: `1px solid ${theme.colors.border}`,
        paddingBottom: theme.spacing.md,
      }}
    >
      {/* Header row. The scorer selector lives here rather than in a band of its
          own between the title and the plot: a lone control floating above the
          bars read as unrelated page furniture, which is part of why the panel
          did not register as a single chart. */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: theme.spacing.sm,
          marginBottom: theme.spacing.sm,
        }}
      >
        {/* Title names the panel; the subtitle carries the scope, which now moves
            with the table selection — so the chart has to say which runs it is
            actually showing rather than assert a fixed "last 5". */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeBase,
              fontWeight: 600,
            }}
          >
            {isSelectionDriven ? (
              <FormattedMessage
                defaultMessage="Selected run summary"
                description="Title of the panel summarising how the evaluation runs selected in the table scored against the baseline"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Recent run summary"
                description="Title of the panel summarising how recent evaluation runs scored against the baseline"
              />
            )}
          </h3>
          {!isCollapsed && (
            <Typography.Hint css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {/* Four separate descriptors rather than one assembled from parts:
                  the message ids are extracted statically at build time, so a
                  descriptor chosen at runtime arrives without an id and throws. */}
              {isSelectionDriven && baselineRun ? (
                <FormattedMessage
                  defaultMessage="{count, plural, one {# selected run} other {# selected runs}} compared with baseline {baselineName}"
                  description="Subtitle naming how many table-selected runs the chart plots and which run is the baseline"
                  values={{ count: chartedRuns.length, baselineName: baselineRun.info.runName }}
                />
              ) : isSelectionDriven ? (
                <FormattedMessage
                  defaultMessage="{count, plural, one {# selected run} other {# selected runs}}"
                  description="Subtitle naming how many table-selected runs the chart plots when no baseline is set"
                  values={{ count: chartedRuns.length }}
                />
              ) : baselineRun ? (
                <FormattedMessage
                  defaultMessage="{count, plural, one {Latest run} other {Latest # runs}} compared with baseline {baselineName} — tick rows to compare specific runs"
                  description="Subtitle naming how many recent runs the chart plots, which run is the baseline, and how to chart a specific selection instead"
                  values={{ count: chartedRuns.length, baselineName: baselineRun.info.runName }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="{count, plural, one {Latest run} other {Latest # runs}}"
                  description="Subtitle naming how many recent runs the chart plots when no baseline is set"
                  values={{ count: chartedRuns.length }}
                />
              )}
            </Typography.Hint>
          )}
        </div>

        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {!isCollapsed && scorerSelector}
          <Button
            componentId="mlflow.eval-runs.benchmark-chart.collapse-button"
            type="tertiary"
            size="small"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? (
              <FormattedMessage defaultMessage="Expand" description="Button to expand the benchmark chart" />
            ) : (
              <FormattedMessage defaultMessage="Collapse" description="Button to collapse the benchmark chart" />
            )}
          </Button>
        </div>
      </div>

      {!isCollapsed && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {/* No baseline message */}
          {!baselineRun && (
            <div
              css={{
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
              }}
            >
              <FormattedMessage
                defaultMessage="Set a baseline to compare against."
                description="Hint message when no baseline is set in benchmark chart"
              />
            </div>
          )}

          {/* An explicit empty state rather than rendering nothing: clearing the
              selector previously left a titled card containing only a dropdown,
              which reads as a broken chart instead of an empty one. Held at the
              chart's own height so clearing and re-selecting doesn't jump the
              page. */}
          {baselineRun && comparableMetricKeys.length === 0 && (
            <div
              css={{
                height: CHART_HEIGHT,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
              }}
            >
              <Empty
                image={<ChartLineIcon />}
                title={
                  <FormattedMessage
                    defaultMessage="No scorers selected"
                    description="Title of the benchmark chart empty state when the scorer selection is empty"
                  />
                }
                description={
                  <FormattedMessage
                    defaultMessage="Choose one or more scorers to compare these runs against the baseline."
                    description="Guidance in the benchmark chart empty state when no scorers are selected"
                  />
                }
                button={
                  <Button
                    componentId="mlflow.eval-runs.benchmark-chart.restore-default-scorers"
                    onClick={() => setUserMetricKeys(null)}
                  >
                    <FormattedMessage
                      defaultMessage="Show default scorers"
                      description="Button in the benchmark chart empty state that restores the default scorer selection"
                    />
                  </Button>
                }
              />
            </div>
          )}

          {/* Silently dropping a scorer would leave the reader counting bars and
              wondering; naming the omission is cheaper than explaining a hole. */}
          {baselineRun && comparableMetricKeys.length < selectedMetricKeys.length && (
            <Typography.Hint>
              <FormattedMessage
                defaultMessage="{count, plural, one {# scorer hidden} other {# scorers hidden}} — not recorded by both the baseline and a compared run, so there is nothing to compare."
                description="Note explaining that some selected scorers are omitted from the benchmark chart because there is no pair of values to compare"
                values={{ count: selectedMetricKeys.length - comparableMetricKeys.length }}
              />
            </Typography.Hint>
          )}

          {/* Same reasoning as the hidden scorers: a selection that is quietly
              truncated looks like a bug in the checkboxes. */}
          {hiddenSelectedRunCount > 0 && (
            <Typography.Hint>
              <FormattedMessage
                defaultMessage="Showing the first {shown} of {total} selected runs — more than that and the bars are too narrow to compare."
                description="Note explaining that the benchmark chart caps how many table-selected runs it plots"
                values={{ shown: chartedRuns.length, total: selectedRuns.length }}
              />
            </Typography.Hint>
          )}

          {/* The bordered surface is what makes this read as a chart rather than
              marks floating on the page: it gives the plot an edge, so the bars
              and the table below are visibly separate objects. */}
          {baselineRun && comparableMetricKeys.length > 0 && (
            <div
              css={{
                height: CHART_HEIGHT,
                border: `1px solid ${theme.colors.borderDecorative}`,
                borderRadius: theme.borders.borderRadiusMd,
                padding: theme.spacing.sm,
                backgroundColor: theme.colors.backgroundPrimary,
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barGap={BAR_GAP}>
                  <CartesianGrid stroke={theme.colors.borderDecorative} vertical={false} />
                  <XAxis
                    dataKey="scorer"
                    tick={{ fontSize: 11, fill: theme.colors.textSecondary }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: theme.colors.textSecondary }}
                    axisLine={false}
                    tickLine={false}
                    width={44}
                    // Anchored at zero deliberately. Cropping the floor to fill
                    // the panel is the truncated-bar-axis lie: it makes a 0.02
                    // gap look like a doubling, and this chart exists to answer
                    // "did it get better" — exactly the judgement a distorted
                    // axis corrupts. Headroom above 1 instead, so the top bar is
                    // not flush with the frame.
                    domain={isBoundedScoreAxis ? [0, 1.05] : undefined}
                  />
                  <Tooltip
                    formatter={tooltipFormatter}
                    cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                    contentStyle={{
                      fontSize: theme.typography.fontSizeSm,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid ${theme.colors.borderDecorative}`,
                    }}
                  />
                  {/* Baseline first so it is the leftmost bar of every cluster —
                      which is also what lets its shape find the cluster's left
                      edge to draw the reference rule from. */}
                  {baselineRun && (
                    <Bar
                      dataKey={baselineRun.info.runUuid}
                      name={baselineLegendName}
                      fill={BASELINE_COLOR}
                      fillOpacity={getOpacity(baselineLegendName)}
                      shape={<BaselineBarShape runUuids={allRunUuids} />}
                    />
                  )}
                  {series.map((entry) => (
                    <Bar
                      key={entry.runUuid}
                      dataKey={entry.runUuid}
                      name={entry.name}
                      fill={entry.color}
                      fillOpacity={getOpacity(entry.name)}
                    />
                  ))}
                  {/* Horizontal, below the plot. A vertical legend on the right
                      took 252px of a 1088px card — a quarter of the width spent
                      on labels, squeezing the bars it was meant to explain. */}
                  <Legend
                    verticalAlign="bottom"
                    iconType="square"
                    wrapperStyle={{ fontSize: 11, maxHeight: 60, overflowY: 'auto' }}
                    onMouseEnter={handleLegendMouseEnter}
                    onMouseLeave={handleLegendMouseLeave}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

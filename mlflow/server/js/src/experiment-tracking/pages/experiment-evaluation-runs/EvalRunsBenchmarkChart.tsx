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
import type { Config, Layout, PlotData } from 'plotly.js';
import React, { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useLocalStorage } from '@databricks/web-shared/hooks';

import { LazyPlot } from '../../components/LazyPlot';
import type { RunEntity } from '../../types';
import { isLowerBetterMetric } from './EvalRunsBaseline.utils';

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
}

/**
 * A single-hue ramp, light → dark, ordered oldest → newest. Because the runs are
 * a sequence rather than unrelated categories, a categorical rainbow would imply
 * they are unrelated; darkening by recency makes "which is latest" readable off
 * the bars without consulting the legend.
 */
const RUN_COLORS = ['#BDD9EC', '#8BBDDE', '#5A9FCF', '#3182BD', '#1F5C8C'];

// Black reads as "the fixed reference" against the blue run ramp.
const BASELINE_COLOR = '#161616';

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: false,
  doubleClick: 'autosize',
  showTips: false,
  // The modebar earns nothing on a fixed 4-category chart — zoom and pan have
  // nothing to reveal — and it renders on top of the legend's first entry.
  displayModeBar: false,
};

/**
 * Fraction of each category slot left empty between scorer groups. Named because
 * the baseline's dashed reference line has to span exactly the cluster it belongs
 * to, which is `1 - BAR_GAP` wide in category units.
 */
const BAR_GAP = 0.2;

/**
 * Plot height. Shared by the chart and its empty state so clearing the scorer
 * selection doesn't move the table below it.
 */
const CHART_HEIGHT = 340;

interface RunWithDataset {
  run: RunEntity;
  datasetName?: string;
}

/**
 * Helper to get the metric value for a run and metric key
 */
const getMetricValue = (run: RunEntity, metricKey: string): number | undefined => {
  const metric = run.data.metrics.find((m) => m.key === metricKey);
  return metric?.value;
};

/**
 * Helper to format metric values to 2 decimal places
 */
const formatMetricValue = (value: number | undefined): string => {
  if (value === undefined) {
    return 'N/A';
  }
  return value.toFixed(2);
};

/**
 * Extract dataset name from run
 */
const getDatasetName = (run: RunEntity): string | undefined => {
  return run.inputs?.datasetInputs?.[0]?.dataset?.name;
};

/**
 * Collapsible benchmark chart comparing the last 5 runs against a baseline.
 * Renders a grouped bar chart where each group represents a scorer (metric key),
 * with bars for the baseline (black) and the last 5 runs.
 */
export const EvalRunsBenchmarkChart = ({ runs, baselineRun, metricKeys }: EvalRunsBenchmarkChartProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

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

  // Get last 5 runs excluding the baseline
  const baselineRunUuid = baselineRun?.info.runUuid;

  const last5Runs = useMemo(() => {
    const filtered = baselineRunUuid ? runs.filter((r) => r.info.runUuid !== baselineRunUuid) : runs;
    // Runs are already sorted newest first by start_time DESC, take first 5
    return filtered.slice(0, 5).reverse(); // Reverse to show oldest→newest left to right
  }, [runs, baselineRunUuid]);

  // A scorer is charted only if it forms an actual comparison: the baseline
  // recorded it AND at least one of the recent runs did. Either half missing
  // leaves a group of one or two bars trailed by a dashed line across empty
  // space, which looks like a rendering fault rather than absent data. The
  // omission is named in a hint below instead.
  const comparableMetricKeys = useMemo(
    () =>
      selectedMetricKeys.filter(
        (key) =>
          baselineRun !== undefined &&
          getMetricValue(baselineRun, key) !== undefined &&
          last5Runs.some((run) => getMetricValue(run, key) !== undefined),
      ),
    [selectedMetricKeys, baselineRun, last5Runs],
  );

  // Build dataset info
  const runsWithDataset = useMemo<RunWithDataset[]>(() => {
    const list: RunWithDataset[] = [];
    if (baselineRun) {
      list.push({
        run: baselineRun,
        datasetName: getDatasetName(baselineRun),
      });
    }
    last5Runs.forEach((run) => {
      list.push({
        run,
        datasetName: getDatasetName(run),
      });
    });
    return list;
  }, [baselineRun, last5Runs]);

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

  // Trigger label: show count summary
  const triggerLabel = useMemo(() => {
    return `${selectedMetricKeys.length} of ${metricKeys.length}`;
  }, [selectedMetricKeys.length, metricKeys.length]);

  const triggerValue = useMemo(
    () => (selectedMetricKeys.length > 0 ? [triggerLabel] : []),
    [selectedMetricKeys.length, triggerLabel],
  );

  // One trace PER RUN, not per metric: plotly groups bars by matching x
  // categories across traces, so a run must be a single trace spanning every
  // scorer for the bars to cluster and for colour to identify the run.
  const plotData = useMemo(() => {
    if (comparableMetricKeys.length === 0) {
      return [];
    }

    const buildTrace = (run: RunEntity, color: string, isBaseline: boolean): Partial<PlotData> => {
      const datasetName = getDatasetName(run);
      return {
        x: comparableMetricKeys,
        y: comparableMetricKeys.map((metricKey) => getMetricValue(run, metricKey) ?? null),
        name: isBaseline
          ? `${run.info.runName} (${formatMessage({
              defaultMessage: 'baseline',
              description: 'Legend suffix marking which series in the benchmark chart is the baseline run',
            })})`
          : (run.info.runName ?? ''),
        type: 'bar',
        marker: { color },
        // The last 5 runs may span different datasets, in which case the bars
        // are not strictly comparable — so the dataset is named on every hover
        // rather than left implied by left-to-right order.
        customdata: comparableMetricKeys.map(() => datasetName ?? '—'),
        hovertemplate: `<b>${run.info.runName}</b><br>%{x}: %{y}<br>dataset: %{customdata}<extra></extra>`,
      } as Partial<PlotData>;
    };

    const traces: Partial<PlotData>[] = [];
    if (baselineRun) {
      traces.push(buildTrace(baselineRun, BASELINE_COLOR, true));
    }
    last5Runs.forEach((run, index) => {
      traces.push(buildTrace(run, RUN_COLORS[index % RUN_COLORS.length], false));
    });
    return traces;
  }, [comparableMetricKeys, baselineRun, last5Runs, formatMessage]);

  // The baseline's dashed reference line is drawn per scorer as a layout shape.
  // Shapes sit on the category axis without becoming legend entries, which a
  // scatter trace per scorer would.
  // Bounded scores get a fixed 0–1 range so a given difference is the same height
  // on every render and across scorers; a metric outside that range (a rate, a
  // count) keeps plotly's autoscaling.
  const isBoundedScoreAxis = useMemo(() => {
    const values = [baselineRun, ...last5Runs]
      .filter((run): run is RunEntity => run !== undefined)
      .flatMap((run) => comparableMetricKeys.map((key) => getMetricValue(run, key)))
      .filter((value): value is number => value !== undefined);
    return values.length > 0 && values.every((value) => value >= 0 && value <= 1);
  }, [baselineRun, last5Runs, comparableMetricKeys]);

  const baselineShapes = useMemo(() => {
    if (!baselineRun) {
      return [];
    }
    return comparableMetricKeys.flatMap((metricKey, index) => {
      const value = getMetricValue(baselineRun, metricKey);
      if (value === undefined) {
        return [];
      }
      return [
        {
          type: 'line' as const,
          xref: 'x' as const,
          yref: 'y' as const,
          // Clamped to the cluster's true width. At the previous ±0.45 the line
          // overhung the bars it annotates and ran into the neighbouring slot,
          // which read as one long rule across the whole chart rather than a
          // per-scorer reference.
          x0: index - (1 - BAR_GAP) / 2,
          x1: index + (1 - BAR_GAP) / 2,
          y0: value,
          y1: value,
          line: { color: BASELINE_COLOR, width: 1.5, dash: 'dash' as const },
        },
      ];
    });
  }, [baselineRun, comparableMetricKeys]);

  // Build layout
  const layoutConfig = useMemo(() => {
    const layout: Partial<Layout> = {
      barmode: 'group',
      // A gap between groups, none within one: the six bars of a scorer touch so
      // the cluster reads as a single shape to compare against its neighbours.
      // At the previous 0.45/0.08 the bars came out 16px wide with more empty
      // space than ink, which is why the panel didn't read as a chart at all.
      bargap: BAR_GAP,
      bargroupgap: 0,
      hovermode: 'closest',
      // Bottom margin carries the tick labels and the horizontal legend beneath them.
      margin: { l: 44, r: 16, t: 8, b: 96 },
      height: CHART_HEIGHT,
      showlegend: true,
      // Horizontal, below the plot. A vertical legend on the right took 252px of
      // a 1088px card — a quarter of the width spent on labels, squeezing the
      // bars it was meant to explain.
      legend: {
        orientation: 'h',
        x: 0,
        y: -0.18,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 11 },
      },
      shapes: baselineShapes,
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      xaxis: {
        type: 'category',
        tickfont: { size: 11, color: theme.colors.textSecondary },
      },
      yaxis: {
        // No rotated axis title: it collides with the tick labels, and on a
        // 0–1 score axis the ticks already say what the numbers are.
        tickfont: { size: 11, color: theme.colors.textSecondary },
        gridcolor: theme.colors.borderDecorative,
        zeroline: false,
        automargin: true,
        // Anchored at zero deliberately. Cropping the floor to fill the panel is
        // the truncated-bar-axis lie: it makes a 0.02 gap look like a doubling,
        // and this chart exists to answer "did it get better" — exactly the
        // judgement a distorted axis corrupts. Headroom above 1 instead, so the
        // top bar is not flush with the frame.
        range: isBoundedScoreAxis ? [0, 1.05] : undefined,
      },
    };
    return layout;
  }, [baselineShapes, theme, isBoundedScoreAxis]);

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
        {/* Title names the panel; the subtitle carries the scope. "Last 5 runs vs
            baseline" was doing both jobs at once and was also inaccurate — the
            selection is `slice(0, 5)`, so with three runs loaded the heading
            claimed five. The subtitle counts what is actually plotted and names
            the baseline, which until now was only discoverable from the legend. */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <h3
            css={{
              margin: 0,
              fontSize: theme.typography.fontSizeBase,
              fontWeight: 600,
            }}
          >
            <FormattedMessage
              defaultMessage="Recent run summary"
              description="Title of the panel summarising how recent evaluation runs scored against the baseline"
            />
          </h3>
          {!isCollapsed && (
            <Typography.Hint css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {baselineRun ? (
                <FormattedMessage
                  defaultMessage="{count, plural, one {Latest run} other {Latest # runs}} compared with baseline {baselineName}"
                  description="Subtitle naming how many recent runs the chart plots and which run is the baseline"
                  values={{ count: last5Runs.length, baselineName: baselineRun.info.runName }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="{count, plural, one {Latest run} other {Latest # runs}}"
                  description="Subtitle naming how many recent runs the chart plots when no baseline is set"
                  values={{ count: last5Runs.length }}
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

      {/* Collapsed state */}
      {isCollapsed && <div />}

      {/* Expanded state */}
      {!isCollapsed && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {/* No baseline message */}
          {!baselineRun && (
            <div
              css={{
                padding: theme.spacing.md,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: 4,
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
                    defaultMessage="Choose one or more scorers to compare the last 5 runs against the baseline."
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
                defaultMessage="{count, plural, one {# scorer hidden} other {# scorers hidden}} — not recorded by both the baseline and a recent run, so there is nothing to compare."
                description="Note explaining that some selected scorers are omitted from the benchmark chart because there is no pair of values to compare"
                values={{ count: selectedMetricKeys.length - comparableMetricKeys.length }}
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
              <LazyPlot
                data={plotData}
                layout={layoutConfig}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

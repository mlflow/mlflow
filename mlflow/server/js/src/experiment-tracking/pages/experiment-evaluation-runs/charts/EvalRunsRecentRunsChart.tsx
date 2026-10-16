import {
  BarChartIcon,
  Button,
  ChartLineIcon,
  ChevronDownIcon,
  DropdownMenu,
  GavelIcon,
  GenericSkeleton,
  SegmentedControlButton,
  SegmentedControlGroup,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  visuallyHidden,
} from '@databricks/design-system';
import { keyBy } from 'lodash';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { createPortal } from 'react-dom';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { Bar, BarChart, Cell, Legend, ReferenceArea, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import type { NumberDomain } from 'recharts';
import { getDefaultChartMetricKeys } from './getDefaultChartMetricKeys';
import { EvalRunsMetricTimelineChart, type EvalRunsMetricTimelinePoint } from './EvalRunsMetricTimelineChart';
import type { RunEntity } from '../../../types';
import { RUNS_COLOR_PALETTE } from '../../../../common/color-palette';
import { useGetExperimentRunColor } from '../../../components/experiment-page/hooks/useExperimentRunColor';
import { useResizeObserver } from '../../../../shared/web-shared/hooks/useResizeObserver';
import { formatEvalRunsAxisValue, formatEvalRunsNumericValue } from '../ExperimentEvaluationRunsNumberFormat';

const ASSESSMENT_COLOR_PALETTE = RUNS_COLOR_PALETTE.slice(0, 9);

const INACTIVE_BAR_OPACITY = 0.55;
const TOOLTIP_VIEWPORT_PADDING = 8;
const TOOLTIP_ANCHOR_GAP = 8;
// Bars can be only a few pixels wide, so a pointer this close to a bar's center still hovers it.
const BAR_HOVER_DISTANCE = 8;
const CHART_Y_AXIS_WIDTH = 44;
const CHART_HORIZONTAL_MARGIN = 16;
const CHART_AREA_HEIGHT = 220;
const COMPACT_CHART_AREA_HEIGHT = 160;
const DEFAULT_ASSESSMENT_LIMIT = 5;
const COMPACT_ASSESSMENT_LIMIT = 3;
const CHRONOLOGY_LABEL_HEIGHT = 20;
const X_AXIS_CATEGORY_PADDING = 8;
const APPROXIMATE_X_AXIS_CHARACTER_WIDTH = 6.5;
const Y_AXIS_DOMAIN_PADDING_RATIO = 0.05;
const Y_AXIS_PADDING = { top: 8, bottom: 8 } as const;
type ChartType = 'line' | 'bar';

/**
 * Keep non-fractional values away from the top edge of the plot. Recharts' automatic domain can
 * place a data maximum on that edge while choosing a lower highest tick (for example, data max 25
 * with a highest tick at 24), which clips the line and the point marker.
 */
export const getEvalRunsYAxisDomain = ([dataMin, dataMax]: NumberDomain): NumberDomain => {
  const dataRange = dataMax - dataMin;
  const padding =
    dataRange > 0
      ? dataRange * Y_AXIS_DOMAIN_PADDING_RATIO
      : Math.max(Math.abs(dataMin), Math.abs(dataMax), 1) * Y_AXIS_DOMAIN_PADDING_RATIO;
  return dataRange > 0 ? [dataMin, dataMax + padding] : [dataMin - padding, dataMax + padding];
};

export const formatEvalRunsXAxisTick = (name: string, chartWidth: number, categoryCount: number): string => {
  if (chartWidth <= 0 || categoryCount <= 0) {
    return name;
  }
  const categoryWidth = (chartWidth - CHART_Y_AXIS_WIDTH - CHART_HORIZONTAL_MARGIN) / categoryCount;
  const characterLimit = Math.max(
    2,
    Math.floor((categoryWidth - X_AXIS_CATEGORY_PADDING) / APPROXIMATE_X_AXIS_CHARACTER_WIDTH),
  );
  return name.length > characterLimit ? `${name.slice(0, characterLimit - 1)}…` : name;
};

const getDatasetName = (run: RunEntity): string | undefined => run.inputs?.datasetInputs?.[0]?.dataset?.name;

// Shared by the bar tooltip and its accessible bar label so the no-value state is consistent.
const NO_VALUE_MESSAGE = defineMessage({
  defaultMessage: 'No value',
  description: 'Eval-runs chart tooltip: shown when a run has no value for an assessment',
});
const BAR_ARIA_LABEL_MESSAGE = defineMessage({
  defaultMessage: '{runName}, {assessment}: {value}',
  description: 'Accessible label for one run and assessment bar in the eval-runs chart',
});

interface ChartRow {
  runUuid: string;
  runName: string;
  startTime: number;
  scores: Record<string, number | null>;
  datasetName?: string;
  createdAt?: number;
}

const formatMetricValue = (value: number | null, isFractional: boolean, noValueText: string): string =>
  value === null
    ? noValueText
    : isFractional
      ? `${formatEvalRunsNumericValue(value * 100)}%`
      : formatEvalRunsNumericValue(value);

interface HoveredBar {
  runUuid: string;
  assessment: string;
  anchorBounds: Pick<DOMRect, 'left' | 'top' | 'bottom' | 'width'>;
}

type XZoomAxis = 'runs' | 'assessments';

interface XZoomRange {
  dataSignature: string;
  startIndex: number;
  endIndex: number;
}

interface XDragSelection {
  startIndex: number;
  endIndex: number;
  startLabel: string | number;
  endLabel: string | number;
}

export const getEvalRunsBarTooltipPosition = (
  anchorBounds: Pick<DOMRect, 'left' | 'top' | 'bottom' | 'width'>,
  tooltipBounds: Pick<DOMRect, 'width' | 'height'>,
  viewportWidth: number,
  viewportHeight: number,
) => {
  const centeredLeft = anchorBounds.left + (anchorBounds.width - tooltipBounds.width) / 2;
  const maximumLeft = Math.max(
    TOOLTIP_VIEWPORT_PADDING,
    viewportWidth - tooltipBounds.width - TOOLTIP_VIEWPORT_PADDING,
  );
  const left = Math.min(Math.max(centeredLeft, TOOLTIP_VIEWPORT_PADDING), maximumLeft);
  const above = anchorBounds.top - tooltipBounds.height - TOOLTIP_ANCHOR_GAP;
  const below = anchorBounds.bottom + TOOLTIP_ANCHOR_GAP;
  const maximumTop = Math.max(
    TOOLTIP_VIEWPORT_PADDING,
    viewportHeight - tooltipBounds.height - TOOLTIP_VIEWPORT_PADDING,
  );
  if (above >= TOOLTIP_VIEWPORT_PADDING) {
    return { left, top: above };
  }

  // A tooltip below a tall bar covers the run labels at the bottom of the chart.
  const sideTop = Math.min(Math.max(anchorBounds.top, TOOLTIP_VIEWPORT_PADDING), maximumTop);
  const right = anchorBounds.left + anchorBounds.width + TOOLTIP_ANCHOR_GAP;
  if (right <= maximumLeft) {
    return { left: Math.max(right, TOOLTIP_VIEWPORT_PADDING), top: sideTop };
  }
  const sideLeft = anchorBounds.left - tooltipBounds.width - TOOLTIP_ANCHOR_GAP;
  if (sideLeft >= TOOLTIP_VIEWPORT_PADDING) {
    return { left: sideLeft, top: sideTop };
  }
  return { left, top: Math.min(Math.max(below, TOOLTIP_VIEWPORT_PADDING), maximumTop) };
};

/**
 * Compact bar tooltip rendered independently of Recharts' payload state so changing bar opacity does
 * not make the tooltip disappear while the pointer is still over a bar.
 */
const EvalRunsBarTooltip = ({
  runName,
  runColor,
  assessment,
  value,
  datasetName,
  createdAt,
  anchorBounds,
}: {
  runName: string;
  runColor: string;
  assessment: string;
  value: string;
  datasetName?: string;
  createdAt?: number;
  anchorBounds: Pick<DOMRect, 'left' | 'top' | 'bottom' | 'width'>;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number }>();

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!tooltip) {
      return;
    }
    setPosition(
      getEvalRunsBarTooltipPosition(
        anchorBounds,
        tooltip.getBoundingClientRect(),
        window.innerWidth,
        window.innerHeight,
      ),
    );
  }, [anchorBounds]);

  return (
    <div
      ref={tooltipRef}
      data-testid="eval-runs-bar-tooltip"
      aria-hidden
      css={{
        position: 'fixed',
        zIndex: theme.options.zIndexBase + 70,
        left: position?.left ?? 0,
        top: position?.top ?? 0,
        visibility: position ? 'visible' : 'hidden',
        pointerEvents: 'none',
        backgroundColor: theme.colors.backgroundPrimary,
        color: theme.colors.textPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        boxShadow: theme.shadows.lg,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        minWidth: 180,
        maxWidth: `min(360px, calc(100vw - ${TOOLTIP_VIEWPORT_PADDING * 2}px))`,
        overflowWrap: 'break-word',
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <div
          css={{
            paddingBottom: theme.spacing.sm,
            marginBottom: theme.spacing.sm,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          <span css={{ display: 'block', fontSize: theme.typography.fontSizeSm, opacity: 0.7 }}>
            <FormattedMessage defaultMessage="Run" description="Run header in the eval runs chart tooltip" />
          </span>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
            <span
              data-testid="eval-runs-tooltip-run-color"
              style={{ backgroundColor: runColor }}
              css={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                border: `1px solid ${theme.colors.border}`,
                flexShrink: 0,
              }}
            />
            <span
              css={{
                minWidth: 0,
                fontSize: theme.typography.fontSizeSm,
                fontWeight: theme.typography.typographyBoldFontWeight,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {runName}
            </span>
          </div>
        </div>
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <GavelIcon aria-hidden="true" />
          <span css={{ fontSize: theme.typography.fontSizeSm, opacity: 0.7 }}>{assessment}</span>
        </span>
        <span
          css={{
            fontSize: theme.typography.fontSizeMd,
            fontWeight: theme.typography.typographyBoldFontWeight,
          }}
        >
          {value}
        </span>
        {(datasetName || createdAt !== undefined) && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
              marginTop: theme.spacing.sm,
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {datasetName && (
              <div
                css={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: theme.spacing.md,
                }}
              >
                <span css={{ opacity: 0.7 }}>
                  <FormattedMessage
                    defaultMessage="Dataset"
                    description="Eval-runs chart tooltip: label for the run's dataset name"
                  />
                </span>
                <span css={{ textAlign: 'right' }}>{datasetName}</span>
              </div>
            )}
            {createdAt !== undefined && (
              <div
                css={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: theme.spacing.md,
                }}
              >
                <span css={{ opacity: 0.7 }}>
                  <FormattedMessage
                    defaultMessage="Created"
                    description="Eval-runs chart tooltip: label for the run's creation time"
                  />
                </span>
                <span css={{ textAlign: 'right' }}>
                  {intl.formatDate(createdAt, {
                    dateStyle: 'medium',
                    timeStyle: 'short',
                  })}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export interface EvalRunsRecentRunsChartProps {
  runs?: RunEntity[];
  /** Uses the same chart in a denser layout for narrow surfaces such as drawers. */
  compact?: boolean;
  /** UUIDs checked in the table. Checked runs remain in the chart with the other runs by default. */
  selectedRunUuids?: string[];
  /** Focus a run from the chart in the table. */
  onRunClick?: (runUuid: string) => void;
  /** When true, hide loaded runs that are not selected in the table. */
  filterToSelected?: boolean;
  /** True while the initial runs request is loading. */
  isLoading?: boolean;
  /** Assessment columns in the same order as the table. */
  metricKeys: string[];
  /** Assessments selected independently in the chart picker. */
  selectedMetricKeys?: string[];
  onSelectedMetricKeysChange: (keys: string[]) => void;
}

/**
 * Collapsible assessment chart above the evaluation-runs table. Run filters are shared with the
 * table; assessment selection is independent.
 */
export const EvalRunsRecentRunsChart = ({
  runs = [],
  compact = false,
  selectedRunUuids = [],
  onRunClick,
  filterToSelected = false,
  isLoading = false,
  metricKeys,
  selectedMetricKeys,
  onSelectedMetricKeysChange,
}: EvalRunsRecentRunsChartProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [chartType, setChartType] = useState<ChartType>('line');
  const getRunColor = useGetExperimentRunColor();
  const chartAreaRef = useRef<HTMLDivElement>(null);
  const chartWidth = useResizeObserver({ ref: chartAreaRef })?.width ?? 0;
  const [hoveredBar, setHoveredBar] = useState<HoveredBar>();
  const [focusedMetrics, setFocusedMetrics] = useState<string[]>([]);
  const [hoveredMetric, setHoveredMetric] = useState<string>();
  const [xZoomRanges, setXZoomRanges] = useState<Record<XZoomAxis, XZoomRange | undefined>>({
    runs: undefined,
    assessments: undefined,
  });
  const [xDragSelection, setXDragSelection] = useState<XDragSelection>();
  const xDragSelectionRef = useRef<XDragSelection>();
  const xDragMovedRef = useRef(false);
  const suppressNextRunClickRef = useRef(false);
  const chartAreaHeight = compact ? COMPACT_CHART_AREA_HEIGHT : CHART_AREA_HEIGHT;

  const safeKeyByMetric = useMemo(() => {
    const map = new Map<string, string>();
    metricKeys.forEach((key, index) => map.set(key, `m${index}`));
    return map;
  }, [metricKeys]);

  const loadedSelectedRunSet = useMemo(() => {
    const loadedRunUuids = new Set(runs.map((run) => run.info.runUuid));
    return new Set(selectedRunUuids.filter((runUuid) => loadedRunUuids.has(runUuid)));
  }, [runs, selectedRunUuids]);
  const hasLoadedSelection = loadedSelectedRunSet.size > 0;
  const shouldFilterToSelected = filterToSelected && hasLoadedSelection;
  const visibleRuns = useMemo(
    () => (shouldFilterToSelected ? runs.filter((run) => loadedSelectedRunSet.has(run.info.runUuid)) : runs),
    [runs, shouldFilterToSelected, loadedSelectedRunSet],
  );
  const defaultMetricKeys = useMemo(
    () =>
      getDefaultChartMetricKeys(visibleRuns, metricKeys, compact ? COMPACT_ASSESSMENT_LIMIT : DEFAULT_ASSESSMENT_LIMIT),
    [visibleRuns, metricKeys, compact],
  );
  // Keep explicit preferences, but recover when none have values in the displayed runs.
  // An intentional empty selection remains empty.
  const resolvedMetricKeys = useMemo(() => {
    if (selectedMetricKeys === undefined) return defaultMetricKeys;
    if (selectedMetricKeys.length === 0) return selectedMetricKeys;
    const hasSelectedValues = visibleRuns.some((run) =>
      run.data.metrics?.some((metric) => selectedMetricKeys.includes(metric.key) && Number.isFinite(metric.value)),
    );
    return hasSelectedValues || defaultMetricKeys.length === 0 ? selectedMetricKeys : defaultMetricKeys;
  }, [selectedMetricKeys, defaultMetricKeys, visibleRuns]);
  const selectedMetrics = useMemo(
    () => metricKeys.filter((key) => resolvedMetricKeys.includes(key)),
    [metricKeys, resolvedMetricKeys],
  );
  const activeFocusedMetrics = focusedMetrics.filter((metric) => selectedMetrics.includes(metric));
  const activeHoveredMetric =
    hoveredMetric !== undefined && selectedMetrics.includes(hoveredMetric) ? hoveredMetric : undefined;
  const getMetricOpacity = (metric: string) => {
    if (activeFocusedMetrics.length === 0 && activeHoveredMetric === undefined) {
      return 1;
    }
    return activeFocusedMetrics.includes(metric) || activeHoveredMetric === metric ? 1 : INACTIVE_BAR_OPACITY;
  };

  // Retain assignments for hidden assessments so toggling selections never shifts other colors.
  const [coloredMetrics, setColoredMetrics] = useState(selectedMetrics);
  const newMetrics = selectedMetrics.filter((metric) => !coloredMetrics.includes(metric));
  const colorOrder = newMetrics.length > 0 ? [...coloredMetrics, ...newMetrics] : coloredMetrics;
  if (newMetrics.length > 0) {
    setColoredMetrics(colorOrder);
  }
  const getMetricColor = (metric: string) =>
    ASSESSMENT_COLOR_PALETTE[colorOrder.indexOf(metric) % ASSESSMENT_COLOR_PALETTE.length];

  const chartData = useMemo(() => {
    const rows = visibleRuns.map((run, originalIndex): ChartRow & { originalIndex: number } => {
      const metricsByKey = keyBy(run.data?.metrics, 'key');
      const scores = Object.fromEntries(
        metricKeys.map((key) => {
          const value = metricsByKey[key]?.value;
          return [safeKeyByMetric.get(key) ?? key, typeof value === 'number' && Number.isFinite(value) ? value : null];
        }),
      );
      return {
        runUuid: run.info.runUuid,
        runName: run.info.runName,
        startTime: Number(run.info.startTime),
        scores,
        datasetName: getDatasetName(run),
        createdAt: Number.isFinite(Number(run.info.startTime)) ? Number(run.info.startTime) : undefined,
        originalIndex,
      };
    });

    // Runs without any selected assessment have nothing to plot and must not reserve axis space.
    return rows
      .filter((row) => selectedMetrics.some((metric) => row.scores[safeKeyByMetric.get(metric) ?? metric] !== null))
      .sort((a, b) => a.startTime - b.startTime || a.originalIndex - b.originalIndex);
  }, [visibleRuns, metricKeys, safeKeyByMetric, selectedMetrics]);

  const isFractional = useMemo(() => {
    const numeric = chartData
      .flatMap((row) => selectedMetrics.map((metric) => row.scores[safeKeyByMetric.get(metric) ?? metric]))
      .filter((value): value is number => value !== null);
    return numeric.length > 0 && numeric.every((value) => value >= 0 && value <= 1);
  }, [chartData, selectedMetrics, safeKeyByMetric]);

  const lineChartPoints = useMemo<EvalRunsMetricTimelinePoint[]>(
    () =>
      chartData.map((row) => ({
        runUuid: row.runUuid,
        runName: row.runName,
        timestampMs: row.startTime,
        scores: Object.fromEntries(
          selectedMetrics.map((metric) => [metric, row.scores[safeKeyByMetric.get(metric) ?? metric]]),
        ),
      })),
    [chartData, safeKeyByMetric, selectedMetrics],
  );

  const hasChartableData = chartData.some((row) =>
    selectedMetrics.some((metric) => row.scores[safeKeyByMetric.get(metric) ?? metric] !== null),
  );

  const allAssessmentsLabel = intl.formatMessage({
    defaultMessage: 'All assessments',
    description: 'Eval-runs chart: metric picker option for charting every assessment column',
  });
  const selectedMetricSet = useMemo(() => new Set(selectedMetrics), [selectedMetrics]);
  const allAssessmentsSelected = metricKeys.length > 0 && selectedMetrics.length === metricKeys.length;
  const metricTriggerLabel = allAssessmentsSelected
    ? allAssessmentsLabel
    : selectedMetrics.length === 1
      ? selectedMetrics[0]
      : selectedMetrics.length > 1
        ? intl.formatMessage(
            {
              defaultMessage: '{count} assessments',
              description: 'Eval-runs chart: metric picker trigger when several assessments are selected',
            },
            { count: selectedMetrics.length },
          )
        : intl.formatMessage({
            defaultMessage: 'No assessments',
            description: 'Eval-runs chart: metric picker label when no assessments are selected',
          });

  const toggleMetric = (metric: string) => {
    const nextSelected = new Set([
      ...resolvedMetricKeys,
      ...(selectedMetricKeys ?? []).filter((key) => !metricKeys.includes(key)),
    ]);
    if (nextSelected.has(metric)) {
      nextSelected.delete(metric);
    } else {
      nextSelected.add(metric);
    }
    onSelectedMetricKeysChange(Array.from(nextSelected));
  };

  const activeHoveredBar =
    chartType === 'bar' &&
    hoveredBar !== undefined &&
    selectedMetrics.includes(hoveredBar.assessment) &&
    chartData.some(
      (row) =>
        row.runUuid === hoveredBar.runUuid &&
        row.scores[safeKeyByMetric.get(hoveredBar.assessment) ?? hoveredBar.assessment] !== null,
    )
      ? hoveredBar
      : undefined;
  const activeHoveredRun = activeHoveredBar
    ? chartData.find((row) => row.runUuid === activeHoveredBar.runUuid)
    : undefined;
  const activeHoveredValue =
    activeHoveredRun && activeHoveredBar
      ? (activeHoveredRun.scores[safeKeyByMetric.get(activeHoveredBar.assessment) ?? activeHoveredBar.assessment] ??
        null)
      : null;

  useEffect(() => {
    if (!activeHoveredBar) {
      return;
    }
    const clearHoveredBar = () => setHoveredBar(undefined);
    window.addEventListener('resize', clearHoveredBar);
    window.addEventListener('scroll', clearHoveredBar, true);
    return () => {
      window.removeEventListener('resize', clearHoveredBar);
      window.removeEventListener('scroll', clearHoveredBar, true);
    };
  }, [activeHoveredBar]);

  const hoverBar = (runUuid: string, assessment: string, event: MouseEvent<SVGElement>) => {
    const { left, top, bottom, width } = event.currentTarget.getBoundingClientRect();
    setHoveredBar({
      runUuid,
      assessment,
      anchorBounds: { left, top, bottom, width },
    });
  };
  const hoverNearestBar = (event: MouseEvent<HTMLDivElement>) => {
    let nearest: { element: SVGElement; bounds: DOMRect; distance: number } | undefined;
    for (const element of chartAreaRef.current?.querySelectorAll<SVGElement>('[data-eval-runs-chart-bar]') ?? []) {
      const bounds = element.getBoundingClientRect();
      const distance = Math.abs(bounds.left + bounds.width / 2 - event.clientX);
      if (
        distance <= BAR_HOVER_DISTANCE &&
        event.clientY <= bounds.bottom &&
        (!nearest || distance < nearest.distance)
      ) {
        nearest = { element, bounds, distance };
      }
    }
    const runUuid = nearest?.element.dataset['runUuid'];
    const assessment = nearest?.element.dataset['assessment'];
    if (!nearest || !runUuid || !assessment) {
      setHoveredBar(undefined);
      return;
    }
    if (hoveredBar?.runUuid === runUuid && hoveredBar.assessment === assessment) {
      return;
    }
    const { left, top, bottom, width } = nearest.bounds;
    setHoveredBar({ runUuid, assessment, anchorBounds: { left, top, bottom, width } });
  };
  const getBarOpacity = (runUuid: string) => {
    if (activeHoveredBar !== undefined) {
      return activeHoveredBar.runUuid === runUuid ? 1 : INACTIVE_BAR_OPACITY;
    }
    if (hasLoadedSelection) {
      return loadedSelectedRunSet.has(runUuid) ? 1 : INACTIVE_BAR_OPACITY;
    }
    return 1;
  };
  const handleRunClick = (runUuid: string) => {
    if (suppressNextRunClickRef.current) {
      suppressNextRunClickRef.current = false;
      return;
    }
    onRunClick?.(runUuid);
  };
  const handleChartClick = (event: MouseEvent<HTMLDivElement>) => {
    if (!(event.target instanceof Element) || event.target.closest('[data-eval-runs-chart-bar]') === null) {
      return;
    }
    // Recharts emits the Cell click with the run payload in the browser. The chart-level fallback
    // also keeps this interaction available in environments where Cell does not forward onClick.
    if (activeHoveredBar) {
      handleRunClick(activeHoveredBar.runUuid);
      return;
    }
    const runUuid = event.target.closest('[data-eval-runs-chart-bar]')?.getAttribute('data-run-uuid');
    if (runUuid) {
      handleRunClick(runUuid);
    }
  };
  const handleLegendClick = (metric: string) => {
    setFocusedMetrics((current) =>
      current.includes(metric) ? current.filter((focusedMetric) => focusedMetric !== metric) : [...current, metric],
    );
  };
  const handleLegendMouseEnter = (metric: string) => setHoveredMetric(metric);
  const handleLegendMouseLeave = () => setHoveredMetric(undefined);

  const getBarAriaLabel = (runName: string, assessment: string, value: number | null) =>
    intl.formatMessage(BAR_ARIA_LABEL_MESSAGE, {
      runName,
      assessment,
      value: formatMetricValue(value, isFractional, intl.formatMessage(NO_VALUE_MESSAGE)),
    });

  const showChronologyAxis = chartType === 'line';
  const lineData = chartData.map((row) => ({ ...row, axisKey: row.runUuid }));
  const barData = selectedMetrics.map((metric) => ({
    axisKey: safeKeyByMetric.get(metric) ?? metric,
    name: metric,
    ...Object.fromEntries(
      chartData.map((row, index) => [`r${index}`, row.scores[safeKeyByMetric.get(metric) ?? metric]]),
    ),
  }));
  const barSeries = chartData.map((row, index) => ({
    key: `r${index}`,
    name: row.runName,
    color: getRunColor(row.runUuid),
    points: selectedMetrics.map((metric) => ({ row, metric })),
  }));
  const xZoomAxis: XZoomAxis = chartType === 'line' ? 'runs' : 'assessments';
  const xAxisData = chartType === 'line' ? lineData : barData;
  const xDataSignature =
    chartType === 'line' ? chartData.map((row) => row.runUuid).join('\0') : selectedMetrics.join('\0');
  const storedXZoomRange = xZoomRanges[xZoomAxis];
  const currentXZoomRange =
    storedXZoomRange?.dataSignature === xDataSignature && storedXZoomRange.startIndex < storedXZoomRange.endIndex
      ? storedXZoomRange
      : undefined;
  const xZoomStartIndex = Math.min(currentXZoomRange?.startIndex ?? 0, Math.max(xAxisData.length - 1, 0));
  const xZoomEndIndex = Math.max(
    xZoomStartIndex,
    Math.min(currentXZoomRange?.endIndex ?? xAxisData.length - 1, Math.max(xAxisData.length - 1, 0)),
  );
  const visibleLineChartPoints = lineChartPoints.slice(xZoomStartIndex, xZoomEndIndex + 1);
  const visibleBarData = barData.slice(xZoomStartIndex, xZoomEndIndex + 1);
  const visibleBarSeries = barSeries.map((series) => ({
    ...series,
    points: series.points.slice(xZoomStartIndex, xZoomEndIndex + 1),
  }));
  const assessmentNameByAxisKey = new Map(
    selectedMetrics.map((metric) => [safeKeyByMetric.get(metric) ?? metric, metric]),
  );
  const getFullXIndex = (activeTooltipIndex: unknown) => {
    const localIndex = Number(activeTooltipIndex);
    const visibleDataLength = xZoomEndIndex - xZoomStartIndex + 1;
    return Number.isInteger(localIndex) && localIndex >= 0 && localIndex < visibleDataLength
      ? xZoomStartIndex + localIndex
      : undefined;
  };
  const beginXDrag = (activeTooltipIndex: unknown, activeLabel: string | number | undefined) => {
    const index = getFullXIndex(activeTooltipIndex);
    if (index === undefined || activeLabel === undefined) return;
    const selection = {
      startIndex: index,
      endIndex: index,
      startLabel: activeLabel,
      endLabel: activeLabel,
    };
    xDragSelectionRef.current = selection;
    xDragMovedRef.current = false;
  };
  const updateXDrag = (activeTooltipIndex: unknown, activeLabel: string | number | undefined) => {
    const index = getFullXIndex(activeTooltipIndex);
    if (index === undefined || activeLabel === undefined) return;
    const selection = xDragSelectionRef.current;
    if (selection && selection.endIndex !== index) {
      xDragMovedRef.current = true;
      const nextSelection = {
        ...selection,
        endIndex: index,
        endLabel: activeLabel,
      };
      xDragSelectionRef.current = nextSelection;
      setXDragSelection(nextSelection);
    }
  };
  const finishXDrag = (activeTooltipIndex: unknown, activeLabel: string | number | undefined) => {
    const endIndex = getFullXIndex(activeTooltipIndex);
    const currentSelection = xDragSelectionRef.current;
    const selection =
      currentSelection && endIndex !== undefined && activeLabel !== undefined
        ? { ...currentSelection, endIndex, endLabel: activeLabel }
        : currentSelection;
    if (selection && xDragMovedRef.current && selection.startIndex !== selection.endIndex) {
      suppressNextRunClickRef.current = true;
      setTimeout(() => {
        suppressNextRunClickRef.current = false;
      }, 0);
      setXZoomRanges((ranges) => ({
        ...ranges,
        [xZoomAxis]: {
          dataSignature: xDataSignature,
          startIndex: Math.min(selection.startIndex, selection.endIndex),
          endIndex: Math.max(selection.startIndex, selection.endIndex),
        },
      }));
    }
    xDragSelectionRef.current = undefined;
    xDragMovedRef.current = false;
    setXDragSelection(undefined);
  };
  const clearXDragSelection = () => {
    xDragSelectionRef.current = undefined;
    xDragMovedRef.current = false;
    setXDragSelection(undefined);
  };
  const resetXZoom = () => {
    setXZoomRanges((ranges) => ({ ...ranges, [xZoomAxis]: undefined }));
    clearXDragSelection();
  };
  // Preserve all loaded runs, with a minimum width per bar instead of squeezing or truncating them.
  const barChartWidth = Math.max(compact ? 320 : 400, visibleBarData.length * (visibleBarSeries.length * 12 + 24) + 60);

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: compact ? theme.spacing.sm : theme.spacing.md,
      }}
    >
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          flexDirection: compact ? 'column' : 'row',
          alignItems: compact ? 'stretch' : 'center',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        }}
      >
        {!compact && chartType === 'bar' && (
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Colors represent runs"
              description="Eval runs chart legend color meaning"
            />
          </Typography.Text>
        )}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: compact ? theme.spacing.xs : theme.spacing.sm,
            flexWrap: 'wrap',
          }}
        >
          {currentXZoomRange && (
            <Typography.Link
              componentId="mlflow.eval-runs.chart.clear-zoom"
              onClick={resetXZoom}
              css={{ fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage defaultMessage="Clear zoom" description="Clear the eval runs chart zoom selection" />
            </Typography.Link>
          )}

          <SegmentedControlGroup
            componentId="mlflow.eval-runs.chart.type"
            name="eval-runs-chart-type"
            value={chartType}
            onChange={(event) => {
              const value = event.target.value;
              if (value === 'line' || value === 'bar') setChartType(value);
              clearXDragSelection();
            }}
          >
            <SegmentedControlButton value="line">
              <Tooltip
                componentId="mlflow.eval-runs.chart.type.line-tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Line chart',
                  description: 'Tooltip for the line chart toggle in the eval-runs chart',
                })}
              >
                <span>
                  <ChartLineIcon />
                  <span css={visuallyHidden}>
                    {intl.formatMessage({
                      defaultMessage: 'Line chart',
                      description: 'Accessible label for the line chart toggle in the eval-runs chart',
                    })}
                  </span>
                </span>
              </Tooltip>
            </SegmentedControlButton>
            <SegmentedControlButton value="bar">
              <Tooltip
                componentId="mlflow.eval-runs.chart.type.bar-tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Bar chart',
                  description: 'Tooltip for the bar chart toggle in the eval-runs chart',
                })}
              >
                <span>
                  <BarChartIcon />
                  <span css={visuallyHidden}>
                    {intl.formatMessage({
                      defaultMessage: 'Bar chart',
                      description: 'Accessible label for the bar chart toggle in the eval-runs chart',
                    })}
                  </span>
                </span>
              </Tooltip>
            </SegmentedControlButton>
          </SegmentedControlGroup>

          <DropdownMenu.Root modal={false}>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.eval-runs.chart.metric-trigger"
                icon={
                  <Tooltip
                    componentId="mlflow.eval-runs.chart.metric-trigger.tooltip"
                    content={intl.formatMessage({
                      defaultMessage: 'Metrics',
                      description: 'Tooltip for the metric icon in the evaluation runs chart',
                    })}
                  >
                    <GavelIcon />
                  </Tooltip>
                }
                disabled={metricKeys.length === 0}
              >
                <span
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                  }}
                >
                  {metricTriggerLabel}
                  <ChevronDownIcon />
                </span>
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              {metricKeys.length > 0 && (
                <DropdownMenu.CheckboxItem
                  componentId="mlflow.eval-runs.chart.metric-all-toggle"
                  checked={allAssessmentsSelected}
                  onSelect={(event) => {
                    event.preventDefault();
                    const hiddenKeys = (selectedMetricKeys ?? []).filter((key) => !metricKeys.includes(key));
                    onSelectedMetricKeysChange(allAssessmentsSelected ? hiddenKeys : [...hiddenKeys, ...metricKeys]);
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  {allAssessmentsLabel}
                </DropdownMenu.CheckboxItem>
              )}
              {metricKeys.length > 0 && <DropdownMenu.Separator />}
              {metricKeys.map((key) => (
                <DropdownMenu.CheckboxItem
                  key={key}
                  componentId="mlflow.eval-runs.chart.metric-toggle"
                  checked={selectedMetricSet.has(key)}
                  onSelect={(event) => {
                    event.preventDefault();
                    toggleMetric(key);
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  <GavelIcon aria-hidden="true" />
                  {key}
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>

      <div
        data-testid="eval-runs-chart-viewport"
        style={{ overflow: 'hidden' }}
        css={{
          height: chartAreaHeight + theme.spacing.md,
          marginTop: theme.spacing.md,
        }}
      >
        <div
          ref={chartAreaRef}
          data-testid="eval-runs-chart-area"
          onClick={handleChartClick}
          onMouseMove={hoverNearestBar}
          onMouseLeave={() => setHoveredBar(undefined)}
          style={{
            width: '100%',
            minWidth: 0,
            userSelect: 'none',
          }}
          css={{
            height: chartAreaHeight,
            minWidth: chartType === 'bar' ? barChartWidth : undefined,
            position: 'relative',
            '.recharts-legend-item': { marginRight: '12px !important' },
            '.recharts-legend-item-text': {
              color: `${theme.colors.textSecondary} !important`,
            },
          }}
        >
          {!isLoading && hasChartableData && showChronologyAxis && (
            <div
              data-testid="eval-runs-chronology-labels"
              css={{
                position: 'absolute',
                right: 8,
                bottom: CHRONOLOGY_LABEL_HEIGHT,
                left: 44,
                zIndex: 1,
                display: 'flex',
                justifyContent: 'space-between',
                color: theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeSm,
                lineHeight: `${CHRONOLOGY_LABEL_HEIGHT}px`,
                pointerEvents: 'none',
              }}
            >
              <span>
                <FormattedMessage defaultMessage="Earliest run" description="Start of the eval runs chart time axis" />
              </span>
              <span>
                <FormattedMessage defaultMessage="Latest run" description="End of the eval runs chart time axis" />
              </span>
            </div>
          )}
          {isLoading ? (
            <GenericSkeleton
              css={{ width: '100%', height: '100%' }}
              label={intl.formatMessage({
                defaultMessage: 'Loading evaluation runs chart',
                description: 'Accessible label for the loading skeleton of the eval-runs chart',
              })}
            />
          ) : hasChartableData ? (
            chartType === 'line' ? (
              <EvalRunsMetricTimelineChart
                points={visibleLineChartPoints}
                metricKeys={selectedMetrics}
                isFractional={isFractional}
                getMetricColor={getMetricColor}
                getRunColor={getRunColor}
                getMetricOpacity={getMetricOpacity}
                selectedRunUuids={Array.from(loadedSelectedRunSet)}
                onRunClick={handleRunClick}
                onLegendClick={handleLegendClick}
                onLegendMouseEnter={handleLegendMouseEnter}
                onLegendMouseLeave={handleLegendMouseLeave}
                nonFractionalYAxisDomain={getEvalRunsYAxisDomain}
                yAxisPadding={Y_AXIS_PADDING}
                onPointMouseDown={beginXDrag}
                onPointMouseMove={updateXDrag}
                onPointMouseUp={finishXDrag}
                onPointerLeave={clearXDragSelection}
                onResetZoom={resetXZoom}
                dragSelection={xDragSelection}
              />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={visibleBarData}
                  margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                  onMouseDown={({ activeTooltipIndex, activeLabel }) => beginXDrag(activeTooltipIndex, activeLabel)}
                  onMouseMove={({ activeTooltipIndex, activeLabel }) => updateXDrag(activeTooltipIndex, activeLabel)}
                  onMouseUp={({ activeTooltipIndex, activeLabel }) => finishXDrag(activeTooltipIndex, activeLabel)}
                  onMouseLeave={clearXDragSelection}
                  onDoubleClick={resetXZoom}
                >
                  <XAxis
                    dataKey="axisKey"
                    tick={{
                      fill: theme.colors.textSecondary,
                      fontSize: theme.typography.fontSizeSm,
                    }}
                    tickLine={false}
                    axisLine={{ stroke: theme.colors.border }}
                    tickFormatter={(axisKey: string) =>
                      formatEvalRunsXAxisTick(
                        assessmentNameByAxisKey.get(axisKey) ?? axisKey,
                        chartWidth,
                        visibleBarData.length,
                      )
                    }
                  />
                  <YAxis
                    domain={
                      isFractional
                        ? [0, 1]
                        : ([min, max]: NumberDomain) => getEvalRunsYAxisDomain([Math.min(0, min), Math.max(0, max)])
                    }
                    ticks={isFractional ? [0, 0.5, 1] : undefined}
                    padding={Y_AXIS_PADDING}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    tick={{
                      fill: theme.colors.textSecondary,
                      fontSize: theme.typography.fontSizeSm,
                    }}
                    tickFormatter={(value: number) =>
                      isFractional ? `${formatEvalRunsAxisValue(value * 100)}%` : formatEvalRunsAxisValue(value)
                    }
                  />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                  {xDragSelection && (
                    <ReferenceArea
                      x1={xDragSelection.startLabel}
                      x2={xDragSelection.endLabel}
                      fill={theme.colors.actionPrimaryBackgroundDefault}
                      fillOpacity={0.16}
                      stroke={theme.colors.actionPrimaryBackgroundDefault}
                      strokeOpacity={0.5}
                    />
                  )}
                  {visibleBarSeries.map((series) => (
                    <Bar
                      key={series.key}
                      dataKey={series.key}
                      name={series.name}
                      fill={series.color}
                      radius={[4, 4, 0, 0]}
                      minPointSize={2}
                      maxBarSize={48}
                      isAnimationActive={false}
                      onMouseEnter={(_data, index, event) => {
                        const point = series.points[index];
                        if (point) {
                          hoverBar(point.row.runUuid, point.metric, event);
                        }
                      }}
                      onMouseLeave={() => setHoveredBar(undefined)}
                    >
                      {series.points.map(({ row, metric }) => (
                        <Cell
                          key={`${row.runUuid}:${metric}`}
                          data-eval-runs-chart-bar
                          data-run-uuid={row.runUuid}
                          data-assessment={metric}
                          role="img"
                          aria-label={getBarAriaLabel(
                            row.runName,
                            metric,
                            row.scores[safeKeyByMetric.get(metric) ?? metric],
                          )}
                          fill={series.color}
                          fillOpacity={getBarOpacity(row.runUuid)}
                          onMouseEnter={(event) => hoverBar(row.runUuid, metric, event)}
                          onMouseLeave={() => setHoveredBar(undefined)}
                          onClick={(event) => {
                            event.stopPropagation();
                            handleRunClick(row.runUuid);
                          }}
                        />
                      ))}
                    </Bar>
                  ))}
                </BarChart>
              </ResponsiveContainer>
            )
          ) : (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: theme.colors.textSecondary,
              }}
            >
              <FormattedMessage
                defaultMessage="No assessment values to chart for the selected runs."
                description="Empty state inside the eval-runs chart when selected assessments have no values"
              />
            </div>
          )}
        </div>
      </div>
      {activeHoveredBar && activeHoveredRun && typeof document !== 'undefined'
        ? createPortal(
            <EvalRunsBarTooltip
              runName={activeHoveredRun.runName}
              runColor={getRunColor(activeHoveredRun.runUuid)}
              assessment={activeHoveredBar.assessment}
              value={formatMetricValue(activeHoveredValue, isFractional, intl.formatMessage(NO_VALUE_MESSAGE))}
              datasetName={activeHoveredRun.datasetName}
              createdAt={activeHoveredRun.createdAt}
              anchorBounds={activeHoveredBar.anchorBounds}
            />,
            document.body,
          )
        : null}
    </div>
  );
};

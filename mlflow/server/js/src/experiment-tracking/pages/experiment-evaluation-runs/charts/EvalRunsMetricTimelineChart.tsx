import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useMemo } from 'react';
import type { ReactElement } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { DotItemDotProps, LegendPayload, NumberDomain } from 'recharts';
import { RUNS_COLOR_PALETTE } from '../../../../common/color-palette';
import { formatEvalRunsNumericValue } from '../ExperimentEvaluationRunsNumberFormat';

const EVAL_RUNS_ASSESSMENT_COLOR_PALETTE = RUNS_COLOR_PALETTE.slice(0, 9);
const INACTIVE_RUN_OPACITY = 0.55;

const formatTooltipLabel = (
  label: unknown,
  payload: ReadonlyArray<{ payload?: unknown }> | undefined,
  getRunColor: ((runUuid: string) => string) | undefined,
  theme: ReturnType<typeof useDesignSystemTheme>['theme'],
  intl: ReturnType<typeof useIntl>,
) => {
  const point: unknown = payload?.[0]?.payload;
  const run = point && typeof point === 'object' ? point : undefined;
  const runName = run && 'runName' in run ? String(run.runName) : String(label);
  const runColor = run && 'runUuid' in run && typeof run.runUuid === 'string' ? getRunColor?.(run.runUuid) : undefined;
  const createdAt =
    run && 'timestampMs' in run && typeof run.timestampMs === 'number' && Number.isFinite(run.timestampMs)
      ? run.timestampMs
      : undefined;
  return (
    <span>
      <span
        css={{
          display: 'block',
          paddingBottom: theme.spacing.sm,
          marginBottom: theme.spacing.sm,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <span css={{ display: 'block', fontSize: theme.typography.fontSizeSm, opacity: 0.7 }}>
          <FormattedMessage defaultMessage="Run" description="Run header in the eval runs chart tooltip" />
        </span>
        <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
          {runColor && (
            <span
              data-testid="eval-runs-tooltip-run-color"
              css={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: runColor,
                border: `1px solid ${theme.colors.border}`,
                flexShrink: 0,
              }}
            />
          )}
          <span css={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {runName}
          </span>
        </span>
      </span>
      {createdAt !== undefined && (
        <span css={{ display: 'block' }}>
          {intl.formatMessage(
            {
              defaultMessage: 'Created: {time}',
              description: 'Run creation time in the evaluation metric timeline tooltip',
            },
            { time: intl.formatDate(createdAt, { dateStyle: 'medium', timeStyle: 'short' }) },
          )}
        </span>
      )}
    </span>
  );
};

export interface EvalRunsMetricTimelinePoint {
  runUuid: string;
  runName: string;
  timestampMs: number;
  scores: Record<string, number | null | undefined>;
}

export interface EvalRunsMetricTimelineChartProps {
  points: readonly EvalRunsMetricTimelinePoint[];
  metricKeys: readonly string[];
  height?: number | string;
  compact?: boolean;
  showLegend?: boolean;
  showYAxis?: boolean;
  yAxisWidth?: number;
  tooltipBelowCursor?: boolean;
  showTooltip?: boolean;
  isFractional?: boolean;
  getMetricColor?: (metric: string, index: number) => string;
  getRunColor?: (runUuid: string) => string;
  getMetricOpacity?: (metric: string) => number;
  getMetricDisplayName?: (metric: string) => string;
  selectedRunUuids?: readonly string[];
  onRunClick?: (runUuid: string) => void;
  onLegendClick?: (metric: string) => void;
  onLegendMouseEnter?: (metric: string) => void;
  onLegendMouseLeave?: () => void;
  nonFractionalYAxisDomain?: (domain: NumberDomain) => NumberDomain;
  yAxisPadding?: { top: number; bottom: number };
  onPointMouseDown?: (activeIndex: unknown, activeLabel: string | number | undefined) => void;
  onPointMouseMove?: (activeIndex: unknown, activeLabel: string | number | undefined) => void;
  onPointMouseUp?: (activeIndex: unknown, activeLabel: string | number | undefined) => void;
  onPointerLeave?: () => void;
  onResetZoom?: () => void;
  dragSelection?: {
    startLabel: string | number;
    endLabel: string | number;
  };
  tooltipContent?: ReactElement;
  getPointOpacity?: (index: number) => number;
  deemphasizeLines?: boolean;
}

const renderEmptyTooltip = () => null;

export const EvalRunsMetricTimelineChart = ({
  points,
  metricKeys,
  height = '100%',
  compact = false,
  showLegend = true,
  showYAxis = true,
  yAxisWidth,
  tooltipBelowCursor = false,
  showTooltip = true,
  isFractional,
  getMetricColor,
  getRunColor,
  getMetricOpacity,
  getMetricDisplayName = (metric) => metric,
  selectedRunUuids,
  onRunClick,
  onLegendClick,
  onLegendMouseEnter,
  onLegendMouseLeave,
  nonFractionalYAxisDomain,
  yAxisPadding,
  onPointMouseDown,
  onPointMouseMove,
  onPointMouseUp,
  onPointerLeave,
  onResetZoom,
  dragSelection,
  tooltipContent,
  getPointOpacity,
  deemphasizeLines = false,
}: EvalRunsMetricTimelineChartProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const resolvedYAxisWidth = yAxisWidth ?? (compact ? theme.spacing.xl + theme.spacing.xs : 44);
  const showCompactLegend = compact && showLegend;
  const safeKeyByMetric = useMemo(() => {
    const map = new Map<string, string>();
    metricKeys.forEach((metric, index) => map.set(metric, `m${index}`));
    return map;
  }, [metricKeys]);
  const selectedRunSet = useMemo(() => new Set(selectedRunUuids ?? []), [selectedRunUuids]);
  const hasSelectedRuns = selectedRunSet.size > 0;
  const resolveMetricOpacity = (metric: string) => getMetricOpacity?.(metric);
  const getLegendMetric = useCallback(
    (entry: LegendPayload): string | undefined => {
      const entryWithName = entry as LegendPayload & { name?: unknown };
      const payload = entry.payload as { dataKey?: unknown; name?: unknown; value?: unknown } | undefined;
      const candidates = [
        entry.dataKey,
        payload?.dataKey,
        entry.value,
        entryWithName.name,
        payload?.name,
        payload?.value,
      ];
      for (const candidate of candidates) {
        if (typeof candidate !== 'string') {
          continue;
        }
        const metric = metricKeys.find((key) => {
          const safeKey = safeKeyByMetric.get(key) ?? key;
          return (
            candidate === `scores.${safeKey}` ||
            candidate === safeKey ||
            candidate === key ||
            candidate === getMetricDisplayName(key)
          );
        });
        if (metric !== undefined) {
          return metric;
        }
      }
      return undefined;
    },
    [getMetricDisplayName, metricKeys, safeKeyByMetric],
  );
  const handleLegendClick = useCallback(
    (entry: LegendPayload) => {
      const metric = getLegendMetric(entry);
      if (metric !== undefined) {
        onLegendClick?.(metric);
      }
    },
    [getLegendMetric, onLegendClick],
  );
  const handleLegendMouseEnter = useCallback(
    (entry: LegendPayload) => {
      const metric = getLegendMetric(entry);
      if (metric !== undefined) {
        onLegendMouseEnter?.(metric);
      }
    },
    [getLegendMetric, onLegendMouseEnter],
  );
  const handleLegendMouseLeave = useCallback(() => onLegendMouseLeave?.(), [onLegendMouseLeave]);
  const chartData = useMemo(
    () =>
      points
        .map((point, originalIndex) => ({
          ...point,
          originalIndex,
          scores: Object.fromEntries(
            metricKeys.map((metric) => {
              const value = point.scores[metric];
              return [
                safeKeyByMetric.get(metric) ?? metric,
                typeof value === 'number' && Number.isFinite(value) ? value : null,
              ];
            }),
          ),
        }))
        .filter((point) => metricKeys.some((metric) => point.scores[safeKeyByMetric.get(metric) ?? metric] !== null))
        .sort((firstPoint, secondPoint) => {
          const hasFirstTimestamp = Number.isFinite(firstPoint.timestampMs);
          const hasSecondTimestamp = Number.isFinite(secondPoint.timestampMs);
          if (hasFirstTimestamp && hasSecondTimestamp) {
            return (
              firstPoint.timestampMs - secondPoint.timestampMs || firstPoint.originalIndex - secondPoint.originalIndex
            );
          }
          if (hasFirstTimestamp !== hasSecondTimestamp) {
            return hasFirstTimestamp ? -1 : 1;
          }
          return firstPoint.originalIndex - secondPoint.originalIndex;
        }),
    [metricKeys, points, safeKeyByMetric],
  );
  const numericValues = chartData
    .flatMap((point) => metricKeys.map((metric) => point.scores[safeKeyByMetric.get(metric) ?? metric]))
    .filter((value): value is number => typeof value === 'number');
  const resolvedIsFractional =
    isFractional ??
    (numericValues.length > 0 && numericValues.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
  const noValueLabel = intl.formatMessage({
    defaultMessage: 'No value',
    description: 'Eval-runs metric timeline tooltip value when a run has no value for a metric',
  });
  const formatTooltipValue = useCallback(
    (value: unknown, name: unknown) => [
      typeof value === 'number'
        ? resolvedIsFractional
          ? `${formatEvalRunsNumericValue(value * 100)}%`
          : formatEvalRunsNumericValue(value)
        : noValueLabel,
      String(name),
    ],
    [noValueLabel, resolvedIsFractional],
  );
  const chartAriaLabel = intl.formatMessage(
    {
      defaultMessage: 'Evaluation metric timeline across {count} {count, plural, one {run} other {runs}}: {metrics}',
      description: 'Accessible summary of the evaluation metric timeline',
    },
    {
      count: chartData.length,
      metrics: intl.formatList(metricKeys.map(getMetricDisplayName), {
        type: 'conjunction',
      }),
    },
  );

  if (chartData.length === 0 || metricKeys.length === 0) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height,
        }}
      >
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="No assessment values to chart"
            description="Empty state for an evaluation metric timeline with no finite values"
          />
        </Typography.Text>
      </div>
    );
  }

  const resolveMetricColor = (metric: string, index: number) =>
    getMetricColor?.(metric, index) ??
    EVAL_RUNS_ASSESSMENT_COLOR_PALETTE[index % EVAL_RUNS_ASSESSMENT_COLOR_PALETTE.length];
  const renderDot = (metric: string, color: string, props: DotItemDotProps) => {
    if (typeof props.value !== 'number' || !Number.isFinite(props.value)) {
      return null;
    }
    const payload =
      typeof props.payload === 'object' && props.payload !== null
        ? (props.payload as { runUuid?: unknown; runName?: unknown })
        : undefined;
    const runUuid = typeof payload?.runUuid === 'string' ? payload.runUuid : undefined;
    const runName = typeof payload?.runName === 'string' ? payload.runName : runUuid;
    const isSelected = runUuid !== undefined && selectedRunSet.has(runUuid);
    return (
      <g
        role={runUuid && onRunClick ? 'button' : undefined}
        tabIndex={runUuid && onRunClick ? 0 : undefined}
        aria-label={
          runUuid && onRunClick
            ? intl.formatMessage(
                {
                  defaultMessage: 'Select run {runName}',
                  description: 'Accessible label for selecting a run by its assessment chart dot',
                },
                { runName: runName ?? runUuid },
              )
            : undefined
        }
        onClick={
          runUuid && onRunClick
            ? (event) => {
                event.stopPropagation();
                onRunClick(runUuid);
              }
            : undefined
        }
        onKeyDown={
          runUuid && onRunClick
            ? (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onRunClick(runUuid);
                }
              }
            : undefined
        }
      >
        {runUuid && onRunClick && <circle cx={props.cx} cy={props.cy} r={12} fill="transparent" />}
        <circle
          cx={props.cx}
          cy={props.cy}
          r={isSelected ? 5 : compact ? 2 : 3}
          fill={color}
          stroke={isSelected ? theme.colors.actionPrimaryBackgroundDefault : color}
          strokeWidth={isSelected ? 2 : 1}
          opacity={
            (resolveMetricOpacity(metric) ?? 1) *
            (isSelected || !hasSelectedRuns ? 1 : INACTIVE_RUN_OPACITY) *
            (getPointOpacity?.(props.index) ?? 1)
          }
          pointerEvents="none"
        />
      </g>
    );
  };
  const timeline = (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={chartData}
        margin={
          compact
            ? {
                top: theme.spacing.xs,
                right: theme.spacing.xs,
                bottom: theme.spacing.xs,
                left: theme.spacing.xs,
              }
            : {
                top: theme.spacing.sm,
                right: theme.spacing.sm,
                left: 0,
                bottom: 0,
              }
        }
        onMouseDown={({ activeTooltipIndex, activeLabel }) => onPointMouseDown?.(activeTooltipIndex, activeLabel)}
        onMouseMove={({ activeTooltipIndex, activeLabel }) => onPointMouseMove?.(activeTooltipIndex, activeLabel)}
        onMouseUp={({ activeTooltipIndex, activeLabel }) => onPointMouseUp?.(activeTooltipIndex, activeLabel)}
        onMouseLeave={onPointerLeave}
        onDoubleClick={onResetZoom}
      >
        <XAxis
          dataKey="runUuid"
          tick={false}
          tickLine={false}
          axisLine={compact ? { stroke: theme.colors.border } : false}
          height={compact ? 1 : 30}
        />
        {showYAxis && (
          <YAxis
            domain={resolvedIsFractional ? [0, 1] : nonFractionalYAxisDomain}
            ticks={resolvedIsFractional ? [0, 0.5, 1] : undefined}
            tickCount={compact ? 3 : undefined}
            interval={compact && resolvedIsFractional ? 0 : undefined}
            padding={yAxisPadding ?? (compact ? { top: theme.spacing.xs, bottom: theme.spacing.xs } : undefined)}
            tickLine={false}
            axisLine={false}
            width={resolvedYAxisWidth}
            tick={{
              fill: theme.colors.textSecondary,
              fontSize: compact ? 9 : theme.typography.fontSizeSm,
            }}
            tickFormatter={(value: number) =>
              resolvedIsFractional ? `${formatEvalRunsNumericValue(value * 100)}%` : formatEvalRunsNumericValue(value)
            }
          />
        )}
        <RechartsTooltip
          content={showTooltip ? tooltipContent : renderEmptyTooltip}
          contentStyle={{
            backgroundColor: theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            maxWidth: 280,
          }}
          itemStyle={{ color: theme.colors.textPrimary }}
          cursor={showTooltip && tooltipContent ? { stroke: theme.colors.actionTertiaryBackgroundHover } : false}
          allowEscapeViewBox={tooltipBelowCursor ? { y: true } : undefined}
          reverseDirection={tooltipBelowCursor ? { y: false } : undefined}
          offset={tooltipBelowCursor ? { x: theme.spacing.xs, y: theme.spacing.sm } : undefined}
          wrapperStyle={tooltipBelowCursor ? { opacity: 1, zIndex: theme.options.zIndexBase + 1 } : undefined}
          formatter={formatTooltipValue}
          labelFormatter={(label, payload) => formatTooltipLabel(label, payload, getRunColor, theme, intl)}
        />
        {!compact && showLegend && (
          <Legend
            iconType="plainline"
            iconSize={16}
            wrapperStyle={{ fontSize: 11 }}
            onClick={onLegendClick ? handleLegendClick : undefined}
            onMouseEnter={onLegendMouseEnter ? handleLegendMouseEnter : undefined}
            onMouseLeave={onLegendMouseLeave ? handleLegendMouseLeave : undefined}
          />
        )}
        {dragSelection && (
          <ReferenceArea
            x1={dragSelection.startLabel}
            x2={dragSelection.endLabel}
            fill={theme.colors.actionPrimaryBackgroundDefault}
            fillOpacity={0.16}
            stroke={theme.colors.actionPrimaryBackgroundDefault}
            strokeOpacity={0.5}
          />
        )}
        {metricKeys.map((metric, index) => {
          const color = resolveMetricColor(metric, index);
          return (
            <Line
              key={metric}
              type="monotone"
              dataKey={`scores.${safeKeyByMetric.get(metric) ?? metric}`}
              name={getMetricDisplayName(metric)}
              stroke={color}
              strokeOpacity={deemphasizeLines ? 0.24 : (resolveMetricOpacity(metric) ?? (compact ? 0.7 : 1))}
              strokeWidth={compact ? 1.5 : 2}
              dot={(props) => renderDot(metric, color, props)}
              activeDot={false}
              connectNulls
              isAnimationActive={false}
            />
          );
        })}
      </LineChart>
    </ResponsiveContainer>
  );

  return (
    <div
      role="img"
      aria-label={chartAriaLabel}
      css={{
        display: showCompactLegend ? 'flex' : 'block',
        flexDirection: showCompactLegend ? 'column' : undefined,
        width: '100%',
        height,
      }}
    >
      {showCompactLegend ? <div css={{ flex: 1, minHeight: 0 }}>{timeline}</div> : timeline}
      {showCompactLegend && (
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
            columnGap: theme.spacing.xs,
            rowGap: 0,
            paddingTop: theme.spacing.xs,
            marginLeft: showYAxis ? resolvedYAxisWidth : 0,
          }}
        >
          {metricKeys.map((metric, index) => {
            const displayName = getMetricDisplayName(metric);
            return (
              <div
                key={metric}
                css={{
                  display: 'flex',
                  minWidth: 0,
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                }}
              >
                <span
                  aria-hidden="true"
                  css={{
                    width: theme.spacing.sm,
                    height: 2,
                    flexShrink: 0,
                    backgroundColor: resolveMetricColor(metric, index),
                  }}
                />
                <span
                  css={{
                    display: 'block',
                    minWidth: 0,
                    overflow: 'hidden',
                    color: theme.colors.textSecondary,
                    fontSize: 10,
                    lineHeight: '12px',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={displayName}
                >
                  {displayName}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

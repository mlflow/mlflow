import { useMemo } from 'react';
import {
  GenericSkeleton,
  SimpleSelect,
  SimpleSelectOption,
  SparkleIcon,
  TableIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, type DotItemDotProps } from 'recharts';
import type { GenAIOverviewMonitorTrendPoint } from '../hooks/useGenAIOverviewMonitorState';
import type { GenAIOverviewChartContext } from '../genAIOverview.types';
import { GENAI_OVERVIEW_TIME_INTERVAL_SECONDS } from '../hooks/useGenAIOverviewState';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import {
  GenAIOverviewChartInteractionContainer,
  type GenAIOverviewChartPopoverContent,
  getGenAIOverviewChartRangeUrl,
  useGenAIOverviewChartInteraction,
} from './GenAIOverviewChartInteraction';

interface MonitorTrendChartRow {
  timestampMs: number;
  dateLabel: string;
  value: number | null;
}

const renderEmptyTooltip = () => null;

const renderMonitorPointDot = (
  { cx, cy, index, payload }: DotItemDotProps,
  latestTimestampMs: number | undefined,
  highlightedIndex: number | undefined,
  color: string,
) => {
  const point = payload as MonitorTrendChartRow | undefined;
  if (
    cx === undefined ||
    cy === undefined ||
    typeof point?.value !== 'number' ||
    !Number.isFinite(point.value) ||
    (highlightedIndex === undefined ? point.timestampMs !== latestTimestampMs : index !== highlightedIndex)
  ) {
    return null;
  }
  return <circle cx={cx} cy={cy} r={3} fill={color} />;
};
export interface GenAIOverviewMonitorTrendChartProps {
  dashboardRoute: string;
  trendByScorerName: Map<string, GenAIOverviewMonitorTrendPoint[]>;
  isLoading?: boolean;
  activeScorerName?: string;
  onAskAssistant?: (context: GenAIOverviewChartContext) => void;
}

export interface GenAIOverviewMonitorScorerSelectorProps {
  scorerNames: string[];
  activeScorerName: string;
  onScorerNameChange: (scorerName: string) => void;
}

export const GenAIOverviewMonitorScorerSelector = ({
  scorerNames,
  activeScorerName,
  onScorerNameChange,
}: GenAIOverviewMonitorScorerSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <SimpleSelect
      componentId="mlflow.genai-overview.monitor.assessment-selector"
      id="mlflow-genai-overview-monitor-assessment-selector"
      label={intl.formatMessage({
        defaultMessage: 'Assessment',
        description: 'Accessible label for the monitoring assessment selector',
      })}
      value={activeScorerName}
      onChange={(event) => onScorerNameChange(event.target.value)}
      isBare
      triggerSize="small"
      maxWidth={theme.spacing.xl * 5}
      contentProps={{ minWidth: theme.spacing.xl * 6, textOverflowMode: 'multiline' }}
    >
      {scorerNames.map((scorerName) => (
        <SimpleSelectOption key={scorerName} value={scorerName}>
          <Typography.Text color="secondary" size="sm">
            {scorerName}
          </Typography.Text>
        </SimpleSelectOption>
      ))}
    </SimpleSelect>
  );
};

export const GenAIOverviewMonitorTrendChart = ({
  dashboardRoute,
  trendByScorerName,
  isLoading = false,
  activeScorerName,
  onAskAssistant,
}: GenAIOverviewMonitorTrendChartProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const chartData = useMemo<MonitorTrendChartRow[]>(
    () =>
      (activeScorerName ? trendByScorerName.get(activeScorerName) : undefined)?.map((point) => ({
        timestampMs: point.timestampMs,
        dateLabel: intl.formatDate(point.timestampMs, {
          month: 'short',
          day: 'numeric',
          timeZone: 'UTC',
        }),
        value: point.value,
      })) ?? [],
    [activeScorerName, intl, trendByScorerName],
  );
  const interaction = useGenAIOverviewChartInteraction(chartData.length);
  const values = chartData
    .map(({ value }) => value)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const latestTimestampMs = [...chartData]
    .reverse()
    .find(({ value }) => typeof value === 'number' && Number.isFinite(value))?.timestampMs;
  const isFractional = values.length > 0 && values.every((value) => value >= 0 && value <= 1);
  const formatValue = (value: number | null | undefined) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return intl.formatMessage({
        defaultMessage: 'No data',
        description: 'Missing value in a monitoring overview chart popover',
      });
    }
    return isFractional
      ? intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 1 })
      : intl.formatNumber(value, { maximumFractionDigits: 3 });
  };
  const formatRangeHeading = (startIndex: number, endIndex: number) => {
    const firstPoint = chartData[startIndex];
    const lastPoint = chartData[endIndex];
    if (!firstPoint || !lastPoint) return '';
    return startIndex === endIndex
      ? intl.formatDate(firstPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' })
      : intl.formatMessage(
          {
            defaultMessage: '{start} – {end}',
            description: 'Selected date range in the monitoring overview chart',
          },
          {
            start: intl.formatDate(firstPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
            end: intl.formatDate(lastPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
          },
        );
  };
  const createPopoverContent = (startIndex: number, endIndex: number): GenAIOverviewChartPopoverContent => {
    const selectedValues = chartData
      .slice(startIndex, endIndex + 1)
      .map(({ value }) => value)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    const isSinglePoint = startIndex === endIndex;
    const displayedValue = isSinglePoint
      ? chartData[startIndex]?.value
      : selectedValues.length > 0
        ? selectedValues.reduce((total, value) => total + value, 0) / selectedValues.length
        : null;
    return {
      heading: formatRangeHeading(startIndex, endIndex),
      subheading: isSinglePoint ? undefined : activeScorerName,
      items: [
        {
          key: 'score',
          label:
            isSinglePoint && activeScorerName
              ? activeScorerName
              : intl.formatMessage({
                  defaultMessage: 'Average',
                  description: 'Average monitoring scorer value in a selected chart range',
                }),
          value: formatValue(displayedValue),
          color: theme.colors.blue400,
        },
      ],
    };
  };
  const hoverContent =
    interaction.hoveredIndex === undefined
      ? undefined
      : createPopoverContent(interaction.hoveredIndex, interaction.hoveredIndex);
  const selectionContent = interaction.selection
    ? createPopoverContent(interaction.selection.startIndex, interaction.selection.endIndex)
    : undefined;
  const selectionActions = interaction.selection
    ? [
        {
          componentId: 'mlflow.genai-overview.monitor-chart.view-dashboard',
          icon: <TableIcon aria-hidden />,
          label:
            interaction.selection.startIndex === interaction.selection.endIndex
              ? intl.formatMessage({
                  defaultMessage: 'View dashboard for this period',
                  description: 'Action for opening the quality dashboard for one selected overview chart period',
                })
              : intl.formatMessage({
                  defaultMessage: 'View dashboard for this range',
                  description: 'Action for opening the quality dashboard for a selected overview chart range',
                }),
          onClick: () => {
            const firstPoint = chartData[interaction.selection?.startIndex ?? 0];
            const lastPoint = chartData[interaction.selection?.endIndex ?? 0];
            if (!firstPoint || !lastPoint) return;
            const route = getGenAIOverviewChartRangeUrl(
              dashboardRoute,
              firstPoint.timestampMs,
              lastPoint.timestampMs + GENAI_OVERVIEW_TIME_INTERVAL_SECONDS * 1000,
            );
            const assessmentHash = activeScorerName ? `#assessment-chart-${encodeURIComponent(activeScorerName)}` : '';
            void navigate(`${route}${assessmentHash}`);
          },
        },
        ...(onAskAssistant && activeScorerName
          ? [
              {
                componentId: 'mlflow.genai-overview.monitor-chart.ask-assistant',
                icon: <SparkleIcon aria-hidden />,
                label: intl.formatMessage({
                  defaultMessage: 'Ask Assistant',
                  description: 'Action for asking Assistant about selected monitoring activity',
                }),
                onClick: () => {
                  const startIndex = interaction.selection?.startIndex ?? 0;
                  const endIndex = interaction.selection?.endIndex ?? 0;
                  const firstPoint = chartData[startIndex];
                  const lastPoint = chartData[endIndex];
                  if (!firstPoint || !lastPoint) return;
                  onAskAssistant({
                    stage: 'monitor',
                    label: intl.formatMessage(
                      {
                        defaultMessage: 'Monitor: {range}',
                        description: 'Context chip for selected monitoring activity',
                      },
                      { range: formatRangeHeading(startIndex, endIndex) },
                    ),
                    startTimeMs: firstPoint.timestampMs,
                    endTimeMsExclusive: lastPoint.timestampMs + GENAI_OVERVIEW_TIME_INTERVAL_SECONDS * 1000,
                    scorerName: activeScorerName,
                    points: chartData.slice(startIndex, endIndex + 1).map(({ timestampMs, value }) => ({
                      timestampMs,
                      value,
                    })),
                  });
                  interaction.clearSelection();
                },
              },
            ]
          : []),
      ]
    : undefined;
  const selectionArea = interaction.visibleRange
    ? {
        start: chartData[interaction.visibleRange.startIndex]?.timestampMs,
        end: chartData[interaction.visibleRange.endIndex]?.timestampMs,
      }
    : undefined;
  const highlightedIndex =
    interaction.hoveredIndex ??
    (interaction.selection?.startIndex === interaction.selection?.endIndex
      ? interaction.selection?.startIndex
      : undefined);
  const isInteracting = interaction.visibleRange !== undefined || interaction.hoveredIndex !== undefined;

  if (!activeScorerName) return <div css={{ height: theme.spacing.xl * 2 }} aria-hidden />;

  return (
    <GenAIOverviewChartInteractionContainer
      interaction={interaction}
      ariaLabel={intl.formatMessage(
        {
          defaultMessage: 'Online scorer trend over the last 30 days: {scorer}. Use arrow keys to select a day.',
          description: 'Accessible label for the interactive overview monitoring trend chart',
        },
        { scorer: activeScorerName },
      )}
      hoverContent={hoverContent}
      selectionContent={selectionContent}
      selectionActions={selectionActions}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          height: theme.spacing.xl * 2,
        }}
      >
        <div
          css={{
            flex: 1,
            position: 'relative',
            zIndex: 1,
            width: '100%',
            minWidth: 0,
            minHeight: 0,
            userSelect: 'none',
            '&::after': {
              content: '""',
              position: 'absolute',
              right: theme.spacing.xs,
              bottom: theme.spacing.xs,
              left: theme.spacing.xs,
              borderBottom: `1px solid ${theme.colors.border}`,
              pointerEvents: 'none',
            },
          }}
        >
          {isLoading ? (
            <GenericSkeleton
              css={{ width: '100%', height: '100%' }}
              label={intl.formatMessage({
                defaultMessage: 'Loading monitoring trend',
                description: 'Accessible label for the monitor trend chart loading state',
              })}
            />
          ) : values.length === 0 ? (
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
              }}
            >
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage
                  defaultMessage="No data for this assessment in the last 30 days"
                  description="Monitor trend chart empty state"
                />
              </Typography.Text>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{
                  top: theme.spacing.xs,
                  right: theme.spacing.xs,
                  bottom: theme.spacing.xs,
                  left: theme.spacing.xs,
                }}
                onMouseDown={({ activeTooltipIndex }) => interaction.onPointMouseDown(activeTooltipIndex)}
                onMouseMove={({ activeTooltipIndex }) => interaction.onPointMouseMove(activeTooltipIndex)}
                onMouseUp={({ activeTooltipIndex }) => interaction.onPointMouseUp(activeTooltipIndex)}
                onMouseLeave={interaction.onPointerLeave}
              >
                <XAxis dataKey="timestampMs" axisLine={false} tick={false} tickLine={false} height={1} />
                <Tooltip content={renderEmptyTooltip} cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }} />
                {selectionArea?.start !== undefined && selectionArea.end !== undefined && (
                  <ReferenceArea
                    x1={selectionArea.start}
                    x2={selectionArea.end}
                    fill={theme.colors.actionPrimaryBackgroundDefault}
                    fillOpacity={0.14}
                    stroke={theme.colors.actionPrimaryBackgroundDefault}
                    strokeOpacity={0.4}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="value"
                  name={activeScorerName}
                  stroke={theme.colors.blue400}
                  strokeOpacity={isInteracting ? 0.24 : 1}
                  strokeWidth={1.5}
                  dot={(props) =>
                    renderMonitorPointDot(props, latestTimestampMs, highlightedIndex, theme.colors.blue400)
                  }
                  activeDot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </GenAIOverviewChartInteractionContainer>
  );
};

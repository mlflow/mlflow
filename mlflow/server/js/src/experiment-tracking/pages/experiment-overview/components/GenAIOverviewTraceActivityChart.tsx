import { useMemo } from 'react';
import { SparkleIcon, TableIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { Bar, Cell, ReferenceArea, XAxis } from 'recharts';
import type { GenAIOverviewActivityPoint, GenAIOverviewChartContext } from '../genAIOverview.types';
import { GENAI_OVERVIEW_TIME_INTERVAL_SECONDS } from '../hooks/useGenAIOverviewState';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { getTracesFilteredUrl } from './OverviewChartComponents';
import { TraceRequestsBarChart } from './TraceRequestsBarChart';
import {
  GenAIOverviewChartInteractionContainer,
  type GenAIOverviewChartPopoverContent,
  getGenAIOverviewChartRangeUrl,
  useGenAIOverviewChartInteraction,
} from './GenAIOverviewChartInteraction';

export interface GenAIOverviewTraceActivityChartProps {
  activity: GenAIOverviewActivityPoint[];
  experimentId: string;
  onAskAssistant?: (context: GenAIOverviewChartContext) => void;
}

export const GenAIOverviewTraceActivityChart = ({
  activity,
  experimentId,
  onAskAssistant,
}: GenAIOverviewTraceActivityChartProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const chartData = useMemo(
    () =>
      activity.map(({ timestampMs, count }) => ({
        name: intl.formatDate(timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
        count,
        timestampMs,
      })),
    [activity, intl],
  );
  const interaction = useGenAIOverviewChartInteraction(chartData.length);
  const formatHeading = (startIndex: number, endIndex: number) => {
    const start = chartData[startIndex];
    const end = chartData[endIndex];
    if (!start || !end) return '';
    return startIndex === endIndex
      ? intl.formatDate(start.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' })
      : intl.formatMessage(
          {
            defaultMessage: '{start} – {end}',
            description: 'Selected date range in the trace activity chart',
          },
          {
            start: intl.formatDate(start.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
            end: intl.formatDate(end.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
          },
        );
  };
  const createPopoverContent = (startIndex: number, endIndex: number): GenAIOverviewChartPopoverContent => ({
    heading: formatHeading(startIndex, endIndex),
    items: [
      {
        key: 'traces',
        label: intl.formatMessage({ defaultMessage: 'Traces', description: 'Trace activity chart tooltip label' }),
        value: chartData
          .slice(startIndex, endIndex + 1)
          .reduce((total, point) => total + point.count, 0)
          .toLocaleString(),
        color: theme.colors.blue400,
      },
    ],
  });
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
          componentId: 'mlflow.genai-overview.trace-chart.view-traces',
          icon: <TableIcon aria-hidden />,
          label:
            interaction.selection.startIndex === interaction.selection.endIndex
              ? intl.formatMessage({
                  defaultMessage: 'View traces for this period',
                  description: 'Action for opening traces from one selected overview chart period',
                })
              : intl.formatMessage({
                  defaultMessage: 'View traces for this range',
                  description: 'Action for opening traces from a selected overview chart range',
                }),
          onClick: () => {
            const firstPoint = chartData[interaction.selection?.startIndex ?? 0];
            const lastPoint = chartData[interaction.selection?.endIndex ?? 0];
            if (!firstPoint || !lastPoint) return;
            const baseRoute = getTracesFilteredUrl(experimentId);
            void navigate(
              getGenAIOverviewChartRangeUrl(
                baseRoute,
                firstPoint.timestampMs,
                lastPoint.timestampMs + GENAI_OVERVIEW_TIME_INTERVAL_SECONDS * 1000,
              ),
            );
          },
        },
        ...(onAskAssistant
          ? [
              {
                componentId: 'mlflow.genai-overview.trace-chart.ask-assistant',
                icon: <SparkleIcon aria-hidden />,
                label: intl.formatMessage({
                  defaultMessage: 'Ask Assistant',
                  description: 'Action for asking Assistant about selected trace activity',
                }),
                onClick: () => {
                  const startIndex = interaction.selection?.startIndex ?? 0;
                  const endIndex = interaction.selection?.endIndex ?? 0;
                  const firstPoint = chartData[startIndex];
                  const lastPoint = chartData[endIndex];
                  if (!firstPoint || !lastPoint) return;
                  onAskAssistant({
                    stage: 'trace',
                    label: intl.formatMessage(
                      {
                        defaultMessage: 'Trace: {range}',
                        description: 'Context chip for selected trace activity',
                      },
                      { range: formatHeading(startIndex, endIndex) },
                    ),
                    startTimeMs: firstPoint.timestampMs,
                    endTimeMsExclusive: lastPoint.timestampMs + GENAI_OVERVIEW_TIME_INTERVAL_SECONDS * 1000,
                    traceCount: chartData
                      .slice(startIndex, endIndex + 1)
                      .reduce((total, point) => total + point.count, 0),
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

  return (
    <GenAIOverviewChartInteractionContainer
      interaction={interaction}
      ariaLabel={intl.formatMessage({
        defaultMessage: 'Trace activity. Use arrow keys to select a day.',
        description: 'Accessible label for the interactive trace activity chart',
      })}
      hoverContent={hoverContent}
      selectionContent={selectionContent}
      selectionActions={selectionActions}
    >
      <TraceRequestsBarChart
        data={chartData}
        height={52}
        margin={{ top: 0, right: theme.spacing.xs, bottom: 0, left: theme.spacing.xs }}
        onMouseDown={({ activeTooltipIndex }) => interaction.onPointMouseDown(activeTooltipIndex)}
        onMouseMove={({ activeTooltipIndex }) => interaction.onPointMouseMove(activeTooltipIndex)}
        onMouseUp={({ activeTooltipIndex }) => interaction.onPointMouseUp(activeTooltipIndex)}
        onMouseLeave={interaction.onPointerLeave}
        bars={
          <Bar dataKey="count" fill={theme.colors.blue400} isAnimationActive={false}>
            {chartData.map((point, index) => (
              <Cell key={point.timestampMs} opacity={interaction.getPointOpacity(index)} />
            ))}
          </Bar>
        }
      >
        <XAxis
          dataKey="timestampMs"
          axisLine={{ stroke: theme.colors.border }}
          tick={false}
          tickLine={false}
          height={1}
        />
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
      </TraceRequestsBarChart>
    </GenAIOverviewChartInteractionContainer>
  );
};

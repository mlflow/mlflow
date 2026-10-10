import { useMemo } from 'react';
import { useIntl } from 'react-intl';
import { SparkleIcon, TableIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Bar, Cell, ReferenceArea, XAxis } from 'recharts';
import type { GenAIOverviewIssueActivityPoint } from '../hooks/useGenAIOverviewAnalyzeState';
import type { GenAIOverviewChartContext } from '../genAIOverview.types';
import { GENAI_OVERVIEW_TIME_INTERVAL_SECONDS } from '../hooks/useGenAIOverviewState';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { TraceRequestsBarChart } from './TraceRequestsBarChart';
import {
  GenAIOverviewChartInteractionContainer,
  type GenAIOverviewChartPopoverContent,
  useGenAIOverviewChartInteraction,
} from './GenAIOverviewChartInteraction';

export interface GenAIOverviewIssueSeverityChartProps {
  activity: GenAIOverviewIssueActivityPoint[];
  issuesRoute: string;
  onAskAssistant?: (context: GenAIOverviewChartContext) => void;
}

export const GenAIOverviewIssueSeverityChart = ({
  activity,
  issuesRoute,
  onAskAssistant,
}: GenAIOverviewIssueSeverityChartProps) => {
  const intl = useIntl();
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const severityLabels = useMemo(
    () => ({
      high: intl.formatMessage({ defaultMessage: 'High', description: 'High issue severity label' }),
      medium: intl.formatMessage({ defaultMessage: 'Medium', description: 'Medium issue severity label' }),
      low: intl.formatMessage({ defaultMessage: 'Low', description: 'Low issue severity label' }),
    }),
    [intl],
  );
  const chartData = useMemo(
    () =>
      activity.map((point) => ({
        name: intl.formatDate(point.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
        timestampMs: point.timestampMs,
        high: point.high,
        medium: point.medium,
        low: point.low,
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
            description: 'Selected date range in the issue activity chart',
          },
          {
            start: intl.formatDate(start.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
            end: intl.formatDate(end.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
          },
        );
  };
  const getIssueCounts = (startIndex: number, endIndex: number) => {
    const selectedPoints = chartData.slice(startIndex, endIndex + 1);
    return {
      low: selectedPoints.reduce((total, point) => total + point.low, 0),
      medium: selectedPoints.reduce((total, point) => total + point.medium, 0),
      high: selectedPoints.reduce((total, point) => total + point.high, 0),
    };
  };
  const createPopoverContent = (startIndex: number, endIndex: number): GenAIOverviewChartPopoverContent => {
    const issueCounts = getIssueCounts(startIndex, endIndex);
    return {
      heading: formatHeading(startIndex, endIndex),
      items: [
        {
          key: 'low',
          label: severityLabels.low,
          value: issueCounts.low.toLocaleString(),
          color: theme.colors.green500,
          labelColor: theme.colors.green500,
        },
        {
          key: 'medium',
          label: severityLabels.medium,
          value: issueCounts.medium.toLocaleString(),
          color: theme.colors.yellow500,
          labelColor: theme.colors.yellow500,
        },
        {
          key: 'high',
          label: severityLabels.high,
          value: issueCounts.high.toLocaleString(),
          color: theme.colors.red500,
          labelColor: theme.colors.red500,
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
          componentId: 'mlflow.genai-overview.issue-chart.view-issues',
          icon: <TableIcon aria-hidden />,
          label: intl.formatMessage({
            defaultMessage: 'View issues',
            description: 'Action for opening the issues page',
          }),
          onClick: () => {
            void navigate(issuesRoute);
          },
        },
        ...(onAskAssistant
          ? [
              {
                componentId: 'mlflow.genai-overview.issue-chart.ask-assistant',
                icon: <SparkleIcon aria-hidden />,
                label: intl.formatMessage({
                  defaultMessage: 'Ask Assistant',
                  description: 'Action for asking Assistant about selected issue activity',
                }),
                onClick: () => {
                  const startIndex = interaction.selection?.startIndex ?? 0;
                  const endIndex = interaction.selection?.endIndex ?? 0;
                  const firstPoint = chartData[startIndex];
                  const lastPoint = chartData[endIndex];
                  if (!firstPoint || !lastPoint) return;
                  onAskAssistant({
                    stage: 'analyze',
                    label: intl.formatMessage(
                      {
                        defaultMessage: 'Analyze: {range}',
                        description: 'Context chip for selected issue activity',
                      },
                      { range: formatHeading(startIndex, endIndex) },
                    ),
                    startTimeMs: firstPoint.timestampMs,
                    endTimeMsExclusive: lastPoint.timestampMs + GENAI_OVERVIEW_TIME_INTERVAL_SECONDS * 1000,
                    issueCounts: getIssueCounts(startIndex, endIndex),
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
  const legendItems = [
    { label: severityLabels.high, color: theme.colors.red500 },
    { label: severityLabels.medium, color: theme.colors.yellow500 },
    { label: severityLabels.low, color: theme.colors.green500 },
  ];
  const renderCells = () =>
    chartData.map((point, index) => <Cell key={point.timestampMs} opacity={interaction.getPointOpacity(index)} />);

  return (
    <GenAIOverviewChartInteractionContainer
      interaction={interaction}
      ariaLabel={intl.formatMessage({
        defaultMessage: 'Issues by severity. Use arrow keys to select a day.',
        description: 'Accessible label for the interactive issue activity chart',
      })}
      hoverContent={hoverContent}
      selectionContent={selectionContent}
      selectionActions={selectionActions}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <TraceRequestsBarChart
          data={chartData}
          height={52}
          margin={{ top: 0, right: theme.spacing.xs, bottom: 0, left: theme.spacing.xs }}
          onMouseDown={({ activeTooltipIndex }) => interaction.onPointMouseDown(activeTooltipIndex)}
          onMouseMove={({ activeTooltipIndex }) => interaction.onPointMouseMove(activeTooltipIndex)}
          onMouseUp={({ activeTooltipIndex }) => interaction.onPointMouseUp(activeTooltipIndex)}
          onMouseLeave={interaction.onPointerLeave}
          bars={
            <>
              <Bar dataKey="low" name={severityLabels.low} stackId="issues" fill={theme.colors.green500}>
                {renderCells()}
              </Bar>
              <Bar dataKey="medium" name={severityLabels.medium} stackId="issues" fill={theme.colors.yellow500}>
                {renderCells()}
              </Bar>
              <Bar dataKey="high" name={severityLabels.high} stackId="issues" fill={theme.colors.red500}>
                {renderCells()}
              </Bar>
            </>
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
        <div css={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: theme.spacing.sm }}>
          {legendItems.map(({ label, color }) => (
            <div key={label} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <span
                aria-hidden="true"
                css={{ width: theme.spacing.sm, height: theme.spacing.sm, borderRadius: 1, backgroundColor: color }}
              />
              <Typography.Text color="secondary" size="sm">
                {label}
              </Typography.Text>
            </div>
          ))}
        </div>
      </div>
    </GenAIOverviewChartInteractionContainer>
  );
};

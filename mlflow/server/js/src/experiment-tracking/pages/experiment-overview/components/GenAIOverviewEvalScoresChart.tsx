import { useMemo } from 'react';
import { SparkleIcon, TableIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { RUNS_COLOR_PALETTE } from '../../../../common/color-palette';
import { EvalRunsMetricTimelineChart } from '../../experiment-evaluation-runs/charts/EvalRunsMetricTimelineChart';
import { formatEvalRunsNumericValue } from '../../experiment-evaluation-runs/ExperimentEvaluationRunsNumberFormat';
import type { GenAIOverviewEvalScorePoint } from '../hooks/useGenAIOverviewEvalState';
import type { GenAIOverviewChartContext } from '../genAIOverview.types';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import {
  GenAIOverviewChartInteractionContainer,
  type GenAIOverviewChartPopoverContent,
  useGenAIOverviewChartInteraction,
} from './GenAIOverviewChartInteraction';

const RECENT_EVAL_RUN_COUNT = 5;
const MAX_ASSISTANT_CONTEXT_SCORE_COUNT = 10;
const EVAL_CHART_VERTICAL_MARGIN_RATIO = 0.1;
const EVAL_RUNS_ASSESSMENT_COLOR_PALETTE = RUNS_COLOR_PALETTE.slice(0, 9);

const getScoreDisplayName = (scoreName: string) => scoreName.replace(/\/mean$/, '');

export interface GenAIOverviewEvalScoresChartProps {
  experimentId: string;
  assessmentScoreNames: string[];
  scorePoints: GenAIOverviewEvalScorePoint[];
  onAskAssistant?: (context: GenAIOverviewChartContext) => void;
}

export const GenAIOverviewEvalScoresChart = ({
  experimentId,
  assessmentScoreNames,
  scorePoints,
  onAskAssistant,
}: GenAIOverviewEvalScoresChartProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const chartHeight = theme.spacing.xl * 2;
  const chartVerticalMargin = chartHeight * EVAL_CHART_VERTICAL_MARGIN_RATIO;
  const recentScorePoints = useMemo(
    () =>
      [...scorePoints]
        .sort((firstPoint, secondPoint) => firstPoint.timestampMs - secondPoint.timestampMs)
        .slice(-RECENT_EVAL_RUN_COUNT),
    [scorePoints],
  );
  const chartScorePoints = useMemo(
    () =>
      recentScorePoints.filter((point) =>
        assessmentScoreNames.some((scoreName) => {
          const value = point.scores[scoreName];
          return typeof value === 'number' && Number.isFinite(value);
        }),
      ),
    [assessmentScoreNames, recentScorePoints],
  );
  const chartScorePointIds = useMemo(() => chartScorePoints.map((point) => point.runUuid), [chartScorePoints]);
  const interaction = useGenAIOverviewChartInteraction(chartScorePoints.length, chartScorePointIds);
  const isFractional = useMemo(() => {
    const numericValues = scorePoints
      .flatMap((point) => assessmentScoreNames.map((scoreName) => point.scores[scoreName]))
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    return numericValues.length > 0 && numericValues.every((value) => value >= 0 && value <= 1);
  }, [assessmentScoreNames, scorePoints]);
  const formatScore = (value: number | null | undefined) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return intl.formatMessage({
        defaultMessage: 'No value',
        description: 'Missing scorer value in the evaluation overview chart popover',
      });
    }
    return isFractional ? `${formatEvalRunsNumericValue(value * 100)}%` : formatEvalRunsNumericValue(value);
  };
  const createPointContent = (index: number): GenAIOverviewChartPopoverContent | undefined => {
    const point = chartScorePoints[index];
    if (!point) return undefined;
    return {
      heading: point.runName,
      subheading: intl.formatDate(point.timestampMs, { dateStyle: 'medium', timeZone: 'UTC' }),
      items: assessmentScoreNames.map((scoreName, scoreIndex) => ({
        key: scoreName,
        label: getScoreDisplayName(scoreName),
        value: formatScore(point.scores[scoreName]),
        color: EVAL_RUNS_ASSESSMENT_COLOR_PALETTE[scoreIndex % EVAL_RUNS_ASSESSMENT_COLOR_PALETTE.length],
      })),
    };
  };
  const createRangeContent = (startIndex: number, endIndex: number): GenAIOverviewChartPopoverContent | undefined => {
    if (startIndex === endIndex) return createPointContent(startIndex);
    const firstPoint = chartScorePoints[startIndex];
    const lastPoint = chartScorePoints[endIndex];
    if (!firstPoint || !lastPoint) return undefined;
    return {
      heading: intl.formatMessage(
        {
          defaultMessage: '{start} – {end}',
          description: 'Selected date range in the evaluation runs overview chart',
        },
        {
          start: intl.formatDate(firstPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
          end: intl.formatDate(lastPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
        },
      ),
      items: [
        {
          key: 'runs',
          label: intl.formatMessage({
            defaultMessage: 'Evaluation runs',
            description: 'Evaluation run count in a selected overview chart range',
          }),
          value: (endIndex - startIndex + 1).toLocaleString(),
        },
      ],
    };
  };
  const formatContextRange = (startIndex: number, endIndex: number) => {
    const firstPoint = chartScorePoints[startIndex];
    const lastPoint = chartScorePoints[endIndex];
    if (!firstPoint || !lastPoint) return '';
    return startIndex === endIndex
      ? intl.formatDate(firstPoint.timestampMs, { dateStyle: 'medium', timeZone: 'UTC' })
      : intl.formatMessage(
          {
            defaultMessage: '{start} – {end}',
            description: 'Selected date range in the evaluation chart context chip',
          },
          {
            start: intl.formatDate(firstPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
            end: intl.formatDate(lastPoint.timestampMs, { month: 'short', day: 'numeric', timeZone: 'UTC' }),
          },
        );
  };
  const hoverContent =
    interaction.hoveredIndex === undefined ? undefined : createPointContent(interaction.hoveredIndex);
  const selectionContent = interaction.selection
    ? createRangeContent(interaction.selection.startIndex, interaction.selection.endIndex)
    : undefined;
  const selectionActions = interaction.selection
    ? [
        {
          componentId: 'mlflow.genai-overview.eval-chart.view-runs',
          icon: <TableIcon aria-hidden />,
          label:
            interaction.selection.startIndex === interaction.selection.endIndex
              ? intl.formatMessage({
                  defaultMessage: 'Open eval run',
                  description: 'Action for opening one selected evaluation run',
                })
              : intl.formatMessage({
                  defaultMessage: 'View evaluation runs',
                  description: 'Action for opening the evaluation runs page',
                }),
          onClick: () => {
            const firstPoint = chartScorePoints[interaction.selection?.startIndex ?? 0];
            if (!firstPoint) return;
            if (interaction.selection?.startIndex === interaction.selection?.endIndex) {
              void navigate(Routes.getIssueDetectionRunDetailsRoute(experimentId, firstPoint.runUuid));
              return;
            }
            const evalRunsRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns);
            void navigate(evalRunsRoute);
          },
        },
        ...(onAskAssistant
          ? [
              {
                componentId: 'mlflow.genai-overview.eval-chart.ask-assistant',
                icon: <SparkleIcon aria-hidden />,
                label: intl.formatMessage({
                  defaultMessage: 'Ask Assistant',
                  description: 'Action for asking Assistant about selected evaluation runs',
                }),
                onClick: () => {
                  const startIndex = interaction.selection?.startIndex ?? 0;
                  const endIndex = interaction.selection?.endIndex ?? 0;
                  const firstPoint = chartScorePoints[startIndex];
                  const lastPoint = chartScorePoints[endIndex];
                  if (!firstPoint || !lastPoint) return;
                  onAskAssistant({
                    stage: 'eval',
                    label: intl.formatMessage(
                      {
                        defaultMessage: 'Eval/Test: {range}',
                        description: 'Context chip for selected evaluation runs',
                      },
                      { range: formatContextRange(startIndex, endIndex) },
                    ),
                    startTimeMs: firstPoint.timestampMs,
                    endTimeMsExclusive: lastPoint.timestampMs + 1,
                    totalScoreCount: assessmentScoreNames.length,
                    runs: chartScorePoints.slice(startIndex, endIndex + 1).map((point) => ({
                      runUuid: point.runUuid,
                      runName: point.runName,
                      timestampMs: point.timestampMs,
                      scores: Object.fromEntries(
                        assessmentScoreNames
                          .slice(0, MAX_ASSISTANT_CONTEXT_SCORE_COUNT)
                          .map((scoreName) => [scoreName, point.scores[scoreName] ?? null]),
                      ),
                    })),
                  });
                  interaction.clearSelection();
                },
              },
            ]
          : []),
      ]
    : undefined;
  const chartSelection = interaction.visibleRange
    ? {
        startLabel: chartScorePoints[interaction.visibleRange.startIndex]?.runUuid,
        endLabel: chartScorePoints[interaction.visibleRange.endIndex]?.runUuid,
      }
    : undefined;

  if (assessmentScoreNames.length === 0) {
    return (
      <div
        css={{
          display: 'flex',
          height: theme.spacing.xl * 2,
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="No evaluation metrics logged"
            description="Evaluation overview chart empty state when runs have no metrics"
          />
        </Typography.Text>
      </div>
    );
  }

  return (
    <GenAIOverviewChartInteractionContainer
      interaction={interaction}
      ariaLabel={intl.formatMessage({
        defaultMessage: 'Evaluation score trends. Use arrow keys to select a run.',
        description: 'Accessible label for the interactive evaluation overview chart',
      })}
      hoverContent={hoverContent}
      selectionContent={selectionContent}
      selectionActions={selectionActions}
    >
      <div
        css={{
          width: '100%',
          height: chartHeight,
          boxSizing: 'border-box',
          padding: `${chartVerticalMargin}px 0`,
        }}
      >
        <EvalRunsMetricTimelineChart
          points={chartScorePoints}
          metricKeys={assessmentScoreNames}
          height="100%"
          compact
          showLegend={false}
          showYAxis={false}
          showTooltip={false}
          isFractional={isFractional}
          getMetricDisplayName={getScoreDisplayName}
          getPointOpacity={interaction.getPointOpacity}
          deemphasizeLines={interaction.visibleRange !== undefined || interaction.hoveredIndex !== undefined}
          onPointMouseDown={(activeIndex) => interaction.onPointMouseDown(activeIndex)}
          onPointMouseMove={(activeIndex) => interaction.onPointMouseMove(activeIndex)}
          onPointMouseUp={(activeIndex) => interaction.onPointMouseUp(activeIndex)}
          onPointerLeave={interaction.onPointerLeave}
          dragSelection={
            chartSelection?.startLabel && chartSelection.endLabel
              ? { startLabel: chartSelection.startLabel, endLabel: chartSelection.endLabel }
              : undefined
          }
        />
      </div>
    </GenAIOverviewChartInteractionContainer>
  );
};

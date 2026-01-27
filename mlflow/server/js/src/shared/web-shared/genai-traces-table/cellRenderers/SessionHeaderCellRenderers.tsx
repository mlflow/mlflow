import React from 'react';

import {
  HoverCard,
  SpeechBubbleIcon,
  TableCell,
  Tag,
  Tooltip,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  FeedbackAssessment,
  TOKEN_USAGE_METADATA_KEY,
  MLFLOW_TRACE_USER_KEY,
  SESSION_ID_METADATA_KEY,
  type ModelTraceInfoV3,
  isFeedbackAssessment,
} from '@databricks/web-shared/model-trace-explorer';

const getSessionIdFromTrace = (trace: ModelTraceInfoV3): string | null => {
  return trace.trace_metadata?.[SESSION_ID_METADATA_KEY] ?? null;
};

import { NullCell } from './NullCell';
import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { StackedComponents } from './StackedComponents';
import { formatDateTime } from './rendererFunctions';
import { EvaluationsReviewAssessmentTag } from '../components/EvaluationsReviewAssessmentTag';
import { RunColorCircle } from '../components/RunColorCircle';
import { formatResponseTitle } from '../GenAiTracesTableBody.utils';
import {
  EXECUTION_DURATION_COLUMN_ID,
  INPUTS_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  SIMULATION_GOAL_COLUMN_ID,
  SIMULATION_PERSONA_COLUMN_ID,
  STATE_COLUMN_ID,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  USER_COLUMN_ID,
} from '../hooks/useTableColumns';
import { TracesTableColumnType, type TracesTableColumn } from '../types';
import { COMPARE_TO_RUN_COLOR, CURRENT_RUN_COLOR } from '../utils/Colors';
import { escapeCssSpecialCharacters } from '../utils/DisplayUtils';
import {
  convertFeedbackAssessmentToRunEvalAssessment,
  getTraceInfoInputs,
  getTraceInfoOutputs,
  MLFLOW_SOURCE_RUN_KEY,
} from '../utils/TraceUtils';
import { compact } from 'lodash';
import { getUniqueValueCountsBySourceId } from '../utils/AggregationUtils';
import { TokenComponent } from './TokensCell';
import { SessionHeaderPassFailAggregatedCell } from './SessionHeaderPassFailAggregatedCell';
import { SessionHeaderNumericAggregatedCell } from './SessionHeaderNumericAggregatedCell';
import { SessionHeaderStringAggregatedCell } from './SessionHeaderStringAggregatedCell';
import { StatusCellRenderer } from './StatusRenderer';
import { calculateSessionDuration } from '../sessions-table/utils';

interface SessionHeaderCellProps {
  column: TracesTableColumn;
  sessionId: string;
  otherSessionId?: string;
  traces: ModelTraceInfoV3[];
  otherTraces?: ModelTraceInfoV3[];
  goal?: string;
  persona?: string;
  onChangeEvaluationId?: (evaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void;
  experimentId: string;
  isComparing?: boolean;
  getRunColor?: (runUuid: string) => string;
}

/**
 * Renders a cell in the session header row based on the column type.
 * Returns null for columns that don't have session-level representations.
 */
export const SessionHeaderCell: React.FC<SessionHeaderCellProps> = ({
  column,
  sessionId,
  otherSessionId,
  traces,
  otherTraces,
  goal,
  persona,
  experimentId,
  isComparing,
  getRunColor,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const firstTrace = traces[0];
  const firstOtherTrace = otherTraces?.[0];

  // Get run colors
  const currentRunUuid = firstTrace?.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
  const otherRunUuid = firstOtherTrace?.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
  const currentRunColor = getRunColor && currentRunUuid ? getRunColor(currentRunUuid) : CURRENT_RUN_COLOR;
  const otherRunColor = getRunColor && otherRunUuid ? getRunColor(otherRunUuid) : COMPARE_TO_RUN_COLOR;

  // Default: render empty cell for columns without session-level data
  let cellContent: React.ReactNode = null;

  // Render specific columns with data from the first trace
  if (column.id === SESSION_COLUMN_ID) {
    // Session ID column - render as a tag with link to session view
    // In comparison mode, show both session IDs stacked (unless they're the same)
    const effectiveOtherSessionId = otherSessionId || (otherTraces?.[0] ? getSessionIdFromTrace(otherTraces[0]) : null);
    const showBothSessionIds = isComparing && effectiveOtherSessionId && effectiveOtherSessionId !== sessionId;

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            sessionId ? (
              <SessionIdLinkWrapper sessionId={sessionId} experimentId={experimentId}>
                <Tag
                  componentId="mlflow.genai-traces-table.session-header-session-id"
                  title={sessionId}
                  css={{ maxWidth: '100%' }}
                >
                  <SpeechBubbleIcon css={{ fontSize: theme.typography.fontSizeBase, marginRight: theme.spacing.xs }} />
                  <Typography.Text ellipsis>{sessionId}</Typography.Text>
                </Tag>
              </SessionIdLinkWrapper>
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            showBothSessionIds ? (
              <SessionIdLinkWrapper sessionId={effectiveOtherSessionId} experimentId={experimentId}>
                <Tag
                  componentId="mlflow.genai-traces-table.session-header-session-id-other"
                  title={effectiveOtherSessionId}
                  css={{ maxWidth: '100%' }}
                >
                  <SpeechBubbleIcon css={{ fontSize: theme.typography.fontSizeBase, marginRight: theme.spacing.xs }} />
                  <Typography.Text ellipsis>{effectiveOtherSessionId}</Typography.Text>
                </Tag>
              </SessionIdLinkWrapper>
            ) : effectiveOtherSessionId ? (
              // Same session ID - show a placeholder to maintain row height consistency
              <div css={{ height: 24 }} />
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = (
        <div css={{ overflow: 'hidden', minWidth: 0, maxWidth: '100%' }}>
          <SessionIdLinkWrapper sessionId={sessionId} experimentId={experimentId}>
            <Tag
              componentId="mlflow.genai-traces-table.session-header-session-id"
              title={sessionId}
              css={{ maxWidth: '100%' }}
            >
              <SpeechBubbleIcon css={{ fontSize: theme.typography.fontSizeBase, marginRight: theme.spacing.xs }} />
              <Typography.Text ellipsis>{sessionId}</Typography.Text>
            </Tag>
          </SessionIdLinkWrapper>
        </div>
      );
    }
  } else if (column.id === INPUTS_COLUMN_ID) {
    // Request/Input column - get from first trace's request_preview
    // In comparison mode, show stacked inputs with run colors
    const inputTitle = firstTrace ? getTraceInfoInputs(firstTrace) : undefined;
    const otherInputTitle = firstOtherTrace ? getTraceInfoInputs(firstOtherTrace) : undefined;

    if (isComparing) {
      cellContent = (
        <div
          css={{
            display: 'flex',
            width: '100%',
            overflow: 'hidden',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.sm,
          }}
        >
          <StackedComponents
            first={
              inputTitle ? (
                <div
                  css={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    minWidth: 0,
                  }}
                  title={inputTitle}
                >
                  {inputTitle}
                </div>
              ) : (
                <NullCell isComparing />
              )
            }
            second={
              otherInputTitle ? (
                <div
                  css={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    minWidth: 0,
                  }}
                  title={otherInputTitle}
                >
                  {otherInputTitle}
                </div>
              ) : (
                <NullCell isComparing />
              )
            }
          />
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, flexShrink: 0 }}>
            <div css={{ display: 'flex' }}>
              {currentRunUuid ? <RunColorCircle color={currentRunColor} /> : <div css={{ width: 12, height: 12 }} />}
            </div>
            <div css={{ display: 'flex' }}>
              {otherRunUuid ? <RunColorCircle color={otherRunColor} /> : <div css={{ width: 12, height: 12 }} />}
            </div>
          </div>
        </div>
      );
    } else {
      cellContent = inputTitle ? (
        <div
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            minWidth: 0,
          }}
          title={inputTitle}
        >
          {inputTitle}
        </div>
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === REQUEST_TIME_COLUMN_ID) {
    // Request time - format timestamp from first trace
    const timestamp = firstTrace?.[REQUEST_TIME_COLUMN_ID];
    const date = timestamp ? new Date(timestamp) : null;
    const otherTimestamp = firstOtherTrace?.[REQUEST_TIME_COLUMN_ID];
    const otherDate = otherTimestamp ? new Date(otherTimestamp) : null;

    const formatOptions = {
      year: 'numeric' as const,
      month: '2-digit' as const,
      day: '2-digit' as const,
      hour: '2-digit' as const,
      minute: '2-digit' as const,
      second: '2-digit' as const,
      hour12: false as const,
    };

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            date ? (
              <Tooltip
                componentId="mlflow.genai-traces-table.session-header-request-time"
                content={date.toLocaleString(navigator.language, { timeZoneName: 'short' })}
              >
                <Typography.Text ellipsis>{formatDateTime(date, intl, formatOptions)}</Typography.Text>
              </Tooltip>
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherDate ? (
              <Tooltip
                componentId="mlflow.genai-traces-table.session-header-request-time-other"
                content={otherDate.toLocaleString(navigator.language, { timeZoneName: 'short' })}
              >
                <Typography.Text ellipsis>{formatDateTime(otherDate, intl, formatOptions)}</Typography.Text>
              </Tooltip>
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = date ? (
        <Tooltip
          componentId="mlflow.genai-traces-table.session-header-request-time"
          content={date.toLocaleString(navigator.language, { timeZoneName: 'short' })}
        >
          <Typography.Text ellipsis>{formatDateTime(date, intl, formatOptions)}</Typography.Text>
        </Tooltip>
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === STATE_COLUMN_ID) {
    // State column - show error/success status
    const lastTrace = traces[traces.length - 1];
    const lastOtherTrace = otherTraces?.[otherTraces.length - 1];

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={lastTrace ? <StatusCellRenderer original={lastTrace} isComparing /> : <NullCell isComparing />}
          second={
            lastOtherTrace ? <StatusCellRenderer original={lastOtherTrace} isComparing /> : <NullCell isComparing />
          }
        />
      );
    } else {
      cellContent = lastTrace ? <StatusCellRenderer original={lastTrace} isComparing={false} /> : <NullCell />;
    }
  } else if (column.id === USER_COLUMN_ID) {
    // User column - get from first trace's metadata or tags
    const value = firstTrace?.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
    const otherValue = firstOtherTrace?.trace_metadata?.[MLFLOW_TRACE_USER_KEY];

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            value ? (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                  overflow: 'hidden',
                  minWidth: 0,
                }}
                title={value}
              >
                <UserIcon css={{ color: theme.colors.textSecondary, fontSize: 16, flexShrink: 0 }} />
                <Typography.Text ellipsis>{value}</Typography.Text>
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherValue ? (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                  overflow: 'hidden',
                  minWidth: 0,
                }}
                title={otherValue}
              >
                <UserIcon css={{ color: theme.colors.textSecondary, fontSize: 16, flexShrink: 0 }} />
                <Typography.Text ellipsis>{otherValue}</Typography.Text>
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = value ? (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            overflow: 'hidden',
            minWidth: 0,
          }}
          title={value}
        >
          <UserIcon css={{ color: theme.colors.textSecondary, fontSize: 16, flexShrink: 0 }} />
          <Typography.Text ellipsis>{value}</Typography.Text>
        </div>
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === RESPONSE_COLUMN_ID) {
    // Response column - get output from the last trace
    const lastTrace = traces[traces.length - 1];
    const lastOtherTrace = otherTraces?.[otherTraces.length - 1];
    const value = lastTrace ? formatResponseTitle(getTraceInfoOutputs(lastTrace)) : undefined;
    const otherValue = lastOtherTrace ? formatResponseTitle(getTraceInfoOutputs(lastOtherTrace)) : undefined;

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            value ? (
              <div
                css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}
                title={value}
              >
                {value}
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherValue ? (
              <div
                css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}
                title={otherValue}
              >
                {otherValue}
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = value ? (
        <div
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            minWidth: 0,
          }}
          title={value}
        >
          {value}
        </div>
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === TOKENS_COLUMN_ID) {
    // Tokens - sum all token counts
    const calculateTokens = (tracesList: ModelTraceInfoV3[]) => {
      let totalTokens = 0;
      let totalInputTokens = 0;
      let totalOutputTokens = 0;

      tracesList.forEach((trace) => {
        const tokenUsage = trace.trace_metadata?.[TOKEN_USAGE_METADATA_KEY];
        try {
          const parsedTokenUsage = tokenUsage ? JSON.parse(tokenUsage) : {};
          totalTokens += parsedTokenUsage.total_tokens || 0;
          totalInputTokens += parsedTokenUsage.input_tokens || 0;
          totalOutputTokens += parsedTokenUsage.output_tokens || 0;
        } catch {
          // Skip invalid token data
        }
      });

      return { totalTokens, totalInputTokens, totalOutputTokens };
    };

    const currentTokens = traces.length > 0 ? calculateTokens(traces) : null;
    const otherTokens = otherTraces && otherTraces.length > 0 ? calculateTokens(otherTraces) : null;

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            currentTokens ? (
              <TokenComponent
                inputTokens={currentTokens.totalInputTokens}
                outputTokens={currentTokens.totalOutputTokens}
                totalTokens={currentTokens.totalTokens}
                isComparing={false}
              />
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherTokens ? (
              <TokenComponent
                inputTokens={otherTokens.totalInputTokens}
                outputTokens={otherTokens.totalOutputTokens}
                totalTokens={otherTokens.totalTokens}
                isComparing={false}
              />
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = currentTokens ? (
        <TokenComponent
          inputTokens={currentTokens.totalInputTokens}
          outputTokens={currentTokens.totalOutputTokens}
          totalTokens={currentTokens.totalTokens}
          isComparing={false}
        />
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === EXECUTION_DURATION_COLUMN_ID) {
    // Duration - sum all execution durations
    const duration = traces.length > 0 ? calculateSessionDuration(traces) : null;
    const otherDuration = otherTraces && otherTraces.length > 0 ? calculateSessionDuration(otherTraces) : null;

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            duration ? (
              <div css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={duration}>
                {duration}
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherDuration ? (
              <div css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={otherDuration}>
                {otherDuration}
              </div>
            ) : (
              <NullCell isComparing />
            )
          }
        />
      );
    } else {
      cellContent = duration ? (
        <div css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={duration}>
          {duration}
        </div>
      ) : (
        <NullCell />
      );
    }
  } else if (column.id === SIMULATION_GOAL_COLUMN_ID) {
    // Goal column - show the simulation goal (same for matched sessions, so show once)
    // Goal should be the same across matched sessions, so just show one value
    cellContent = goal ? (
      <div
        css={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          minWidth: 0,
        }}
        title={goal}
      >
        {goal}
      </div>
    ) : (
      <NullCell />
    );
  } else if (column.id === SIMULATION_PERSONA_COLUMN_ID) {
    // Persona column - show the simulation persona (same for matched sessions, so show once)
    // Persona should be the same across matched sessions, so just show one value
    cellContent = persona ? (
      <div
        css={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          minWidth: 0,
        }}
        title={persona}
      >
        {persona}
      </div>
    ) : (
      <NullCell />
    );
  } else if (
    column.type === TracesTableColumnType.ASSESSMENT &&
    column.assessmentInfo &&
    column.assessmentInfo?.isSessionLevelAssessment
  ) {
    // Session-level assessment column - find the assessment with session metadata
    const assessmentInfo = column.assessmentInfo;

    const getSessionLevelAssessments = (tracesList: ModelTraceInfoV3[]) => {
      const allFeedback = tracesList.flatMap((trace) =>
        compact(
          trace.assessments?.filter(
            (a): a is FeedbackAssessment =>
              Boolean(a.metadata?.[ASSESSMENT_SESSION_METADATA_KEY]) && isFeedbackAssessment(a),
          ),
        ),
      );
      const entries = allFeedback.map(convertFeedbackAssessmentToRunEvalAssessment);
      return getUniqueValueCountsBySourceId(assessmentInfo, entries);
    };

    const renderAssessmentTags = (uniqueValueCounts: ReturnType<typeof getUniqueValueCountsBySourceId>) => (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {uniqueValueCounts.map((uniqueValueCount) => (
          <EvaluationsReviewAssessmentTag
            key={`tag_${uniqueValueCount.latestAssessment.name}_${uniqueValueCount.value}`}
            showRationaleInTooltip
            disableJudgeTypeIcon
            hideAssessmentName
            assessment={uniqueValueCount.latestAssessment}
            isRootCauseAssessment={false}
            assessmentInfo={assessmentInfo}
            type="value"
            count={uniqueValueCount.count}
          />
        ))}
      </div>
    );

    const currentUniqueValueCounts = traces.length > 0 ? getSessionLevelAssessments(traces) : [];
    const otherUniqueValueCounts = otherTraces && otherTraces.length > 0 ? getSessionLevelAssessments(otherTraces) : [];

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={
            currentUniqueValueCounts.length > 0 ? (
              renderAssessmentTags(currentUniqueValueCounts)
            ) : (
              <NullCell isComparing />
            )
          }
          second={
            otherUniqueValueCounts.length > 0 ? renderAssessmentTags(otherUniqueValueCounts) : <NullCell isComparing />
          }
        />
      );
    } else {
      cellContent = currentUniqueValueCounts.length > 0 ? renderAssessmentTags(currentUniqueValueCounts) : <NullCell />;
    }
  } else if (
    column.type === TracesTableColumnType.ASSESSMENT &&
    column.assessmentInfo &&
    !column.assessmentInfo?.isSessionLevelAssessment
  ) {
    // Non-session-level assessment column - aggregate values from all traces
    const assessmentInfo = column.assessmentInfo;
    const { dtype } = assessmentInfo;

    const renderAggregatedCell = (tracesList: ModelTraceInfoV3[]) => {
      if (dtype === 'pass-fail' || dtype === 'boolean') {
        return <SessionHeaderPassFailAggregatedCell assessmentInfo={assessmentInfo} traces={tracesList} />;
      } else if (dtype === 'numeric') {
        return <SessionHeaderNumericAggregatedCell assessmentInfo={assessmentInfo} traces={tracesList} />;
      } else if (dtype === 'string' || dtype === 'unknown') {
        return <SessionHeaderStringAggregatedCell assessmentInfo={assessmentInfo} traces={tracesList} />;
      } else {
        return <NullCell isComparing={isComparing} />;
      }
    };

    if (isComparing) {
      cellContent = (
        <StackedComponents
          first={traces.length > 0 ? renderAggregatedCell(traces) : <NullCell isComparing />}
          second={otherTraces && otherTraces.length > 0 ? renderAggregatedCell(otherTraces) : <NullCell isComparing />}
        />
      );
    } else {
      cellContent = traces.length > 0 ? renderAggregatedCell(traces) : <NullCell />;
    }
  }

  return (
    <TableCell
      key={column.id}
      wrapContent={false}
      style={{
        flex: `1 1 var(--col-${escapeCssSpecialCharacters(column.id)}-size)`,
      }}
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
      }}
    >
      <div css={{ display: 'flex', overflow: 'hidden', minWidth: 0, flex: 1 }}>{cellContent}</div>
    </TableCell>
  );
};

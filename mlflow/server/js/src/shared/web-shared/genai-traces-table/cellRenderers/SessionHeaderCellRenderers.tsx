import React from 'react';

import type { ThemeType } from '@databricks/design-system';
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
import type { IntlShape } from '@databricks/i18n';
import {
  getAssessmentValue,
  isSessionLevelAssessment,
  TOKEN_USAGE_METADATA_KEY,
  type Assessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';

import { EvaluationsReviewAssessmentTag, isAssessmentPassing } from '../components/EvaluationsReviewAssessmentTag';
import type { AssessmentInfo } from '../types';
import { TracesTableColumnType } from '../types';
import { getAssessmentValueBarBackgroundColor } from '../utils/Colors';

import { NullCell } from './NullCell';
import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { formatDateTime } from './rendererFunctions';
import { formatResponseTitle } from '../GenAiTracesTableBody.utils';
import {
  EXECUTION_DURATION_COLUMN_ID,
  INPUTS_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  USER_COLUMN_ID,
} from '../hooks/useTableColumns';
import type { TracesTableColumn } from '../types';
import { escapeCssSpecialCharacters } from '../utils/DisplayUtils';
import { getTraceInfoOutputs } from '../utils/TraceUtils';

interface SessionHeaderCellProps {
  column: TracesTableColumn;
  sessionId: string;
  traces: ModelTraceInfoV3[];
  onChangeEvaluationId?: (evaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void;
  theme: ThemeType;
  intl: IntlShape;
  experimentId: string;
}

/**
 * Renders a cell in the session header row based on the column type.
 * Returns null for columns that don't have session-level representations.
 */
export const SessionHeaderCell: React.FC<SessionHeaderCellProps> = ({
  column,
  sessionId,
  traces,
  theme,
  intl,
  experimentId,
}) => {
  const firstTrace = traces[0];

  // Default: render empty cell for columns without session-level data
  let cellContent: React.ReactNode = null;

  // Render specific columns with data from the first trace
  if (column.id === TRACE_ID_COLUMN_ID) {
    // Session ID column - render as a tag with link to session view
    cellContent = (
      <div css={{ overflow: 'hidden', minWidth: 0, maxWidth: '100%' }}>
        <SessionIdLinkWrapper sessionId={sessionId} experimentId={experimentId}>
          <Tag
            componentId="mlflow.genai-traces-table.session-header-session-id"
            title={sessionId}
            css={{ maxWidth: '100%' }}
          >
            <SpeechBubbleIcon css={{ fontSize: theme.typography.fontSizeBase, marginRight: theme.spacing.xs }} />
            <span
              css={{
                display: 'inline-flex',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                alignItems: 'center',
                minWidth: 0,
                flexShrink: 1,
              }}
            >
              {sessionId}
            </span>
          </Tag>
        </SessionIdLinkWrapper>
      </div>
    );
  } else if (column.id === INPUTS_COLUMN_ID && firstTrace) {
    // Request/Input column - get from first trace's request_preview
    const inputTitle = firstTrace.request_preview || '';
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
  } else if (column.id === REQUEST_TIME_COLUMN_ID && firstTrace) {
    // Request time - format timestamp from first trace
    const timestamp = firstTrace[REQUEST_TIME_COLUMN_ID];
    const date = timestamp ? new Date(timestamp) : null;
    cellContent = date ? (
      <Tooltip
        componentId="mlflow.genai-traces-table.session-header-request-time"
        content={date.toLocaleString(navigator.language, { timeZoneName: 'short' })}
      >
        <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>
          {formatDateTime(date, intl, {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
          })}
        </span>
      </Tooltip>
    ) : (
      <NullCell />
    );
  } else if (column.id === USER_COLUMN_ID && firstTrace) {
    // User column - get from first trace's metadata or tags
    const value = firstTrace.trace_metadata?.['mlflow.trace.user'] || firstTrace.tags?.['mlflow.user'];
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
        <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>{value}</span>
      </div>
    ) : (
      <NullCell />
    );
  } else if (column.id === RESPONSE_COLUMN_ID && traces.length > 0) {
    // Response column - get output from the last trace
    const lastTrace = traces[traces.length - 1];
    const value = formatResponseTitle(getTraceInfoOutputs(lastTrace));
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
  } else if (column.id === TOKENS_COLUMN_ID && traces.length > 0) {
    // Tokens - sum all token counts
    let totalTokens = 0;
    let totalInputTokens = 0;
    let totalOutputTokens = 0;

    traces.forEach((trace) => {
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

    cellContent =
      totalTokens > 0 ? (
        <HoverCard
          trigger={
            <Tag
              css={{ width: 'fit-content', maxWidth: '100%' }}
              componentId="mlflow.genai-traces-table.session-tokens"
            >
              <span
                css={{
                  display: 'block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {totalTokens}
              </span>
            </Tag>
          }
          content={
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              <div css={{ display: 'flex', flexDirection: 'row' }}>
                <div css={{ width: '35%' }}>
                  <Typography.Text>
                    {intl.formatMessage({
                      defaultMessage: 'Total',
                      description: 'Label for the total tokens in the tooltip for the tokens cell.',
                    })}
                  </Typography.Text>
                </div>
                <div>
                  <Typography.Text color="secondary">{totalTokens}</Typography.Text>
                </div>
              </div>
              {totalInputTokens > 0 && (
                <div css={{ display: 'flex', flexDirection: 'row' }}>
                  <div css={{ width: '35%' }}>
                    <Typography.Text>
                      {intl.formatMessage({
                        defaultMessage: 'Input',
                        description: 'Label for the input tokens in the tooltip for the tokens cell.',
                      })}
                    </Typography.Text>
                  </div>
                  <div>
                    <Typography.Text color="secondary">{totalInputTokens}</Typography.Text>
                  </div>
                </div>
              )}
              {totalOutputTokens > 0 && (
                <div css={{ display: 'flex', flexDirection: 'row' }}>
                  <div css={{ width: '35%' }}>
                    <Typography.Text>
                      {intl.formatMessage({
                        defaultMessage: 'Output',
                        description: 'Label for the output tokens in the tooltip for the tokens cell.',
                      })}
                    </Typography.Text>
                  </div>
                  <div>
                    <Typography.Text color="secondary">{totalOutputTokens}</Typography.Text>
                  </div>
                </div>
              )}
            </div>
          }
        />
      ) : (
        <NullCell />
      );
  } else if (column.type === TracesTableColumnType.ASSESSMENT && column.assessmentInfo && traces.length > 0) {
    // Assessment column - check for session-level assessments or aggregate pass/fail
    const assessmentInfo = column.assessmentInfo as AssessmentInfo;
    const assessmentName = assessmentInfo.name;

    // Collect all assessments matching this name from all traces
    const allAssessments: Assessment[] = [];
    traces.forEach((trace) => {
      const traceAssessments = trace.assessments ?? [];
      traceAssessments.forEach((assessment) => {
        if (assessment.assessment_name === assessmentName) {
          allAssessments.push(assessment);
        }
      });
    });

    // Check for session-level assessments
    const sessionAssessments = allAssessments.filter((a) => isSessionLevelAssessment(a));

    if (sessionAssessments.length > 0) {
      // Display the session-level assessment(s)
      const latestAssessment = sessionAssessments[sessionAssessments.length - 1];
      const value = getAssessmentValue(latestAssessment);

      // Map source type from model-trace-explorer format to genai-traces-table format
      const sourceType = latestAssessment.source?.source_type;
      const mappedSourceType =
        sourceType === 'LLM_JUDGE'
          ? 'AI_JUDGE'
          : sourceType === 'HUMAN'
            ? 'HUMAN'
            : sourceType === 'CODE'
              ? 'CODE'
              : 'AI_JUDGE';

      // Convert timestamp string to number
      const timestampNum = latestAssessment.create_time ? new Date(latestAssessment.create_time).getTime() : null;

      cellContent = (
        <EvaluationsReviewAssessmentTag
          showRationaleInTooltip
          disableJudgeTypeIcon
          hideAssessmentName
          assessment={{
            name: assessmentName,
            rationale: latestAssessment.rationale ?? null,
            source: {
              sourceId: latestAssessment.source?.source_id ?? '',
              sourceType: mappedSourceType,
              metadata: {},
            },
            stringValue: typeof value === 'string' ? value : null,
            booleanValue: typeof value === 'boolean' ? value : null,
            numericValue: typeof value === 'number' ? value : null,
            rootCauseAssessment: null,
            timestamp: timestampNum,
            metadata: {},
          }}
          assessmentInfo={assessmentInfo}
          type="value"
        />
      );
    } else if (assessmentInfo.dtype === 'pass-fail' || assessmentInfo.dtype === 'boolean') {
      // Aggregate pass/fail assessments
      let passCount = 0;
      let totalCount = 0;

      allAssessments.forEach((assessment) => {
        const value = getAssessmentValue(assessment);
        // Skip array values as they're not applicable for pass/fail aggregation
        if (Array.isArray(value)) {
          return;
        }
        const isPassing = isAssessmentPassing(assessmentInfo, value);
        if (isPassing !== undefined) {
          totalCount++;
          if (isPassing) {
            passCount++;
          }
        }
      });

      if (totalCount > 0) {
        const passFraction = passCount / totalCount;
        const failFraction = 1 - passFraction;
        const passColor = getAssessmentValueBarBackgroundColor(theme, assessmentInfo, 'yes', false);
        const failColor = getAssessmentValueBarBackgroundColor(theme, assessmentInfo, 'no', false);

        cellContent = (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              width: '100%',
              minWidth: 0,
            }}
          >
            {/* Stacked bar showing pass (green) and fail (red) proportions */}
            <div
              css={{
                display: 'flex',
                height: 8,
                borderRadius: 4,
                overflow: 'hidden',
                flexShrink: 0,
                width: 40,
              }}
            >
              <div
                style={{
                  width: `${passFraction * 100}%`,
                  backgroundColor: passColor,
                }}
              />
              <div
                style={{
                  width: `${failFraction * 100}%`,
                  backgroundColor: failColor,
                }}
              />
            </div>
            {/* Count label e.g. "3/4 PASS" */}
            <span
              css={{
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
                whiteSpace: 'nowrap',
              }}
            >
              {passCount}/{totalCount}{' '}
              {intl.formatMessage({
                defaultMessage: 'PASS',
                description: 'Label for pass count in session assessment aggregation',
              })}
            </span>
          </div>
        );
      } else {
        cellContent = <NullCell />;
      }
    } else {
      cellContent = <NullCell />;
    }
  }

  return (
    <TableCell
      key={column.id}
      style={{
        flex: `1 1 var(--col-${escapeCssSpecialCharacters(column.id)}-size)`,
      }}
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
      }}
    >
      <div css={{ display: 'flex', overflow: 'hidden', minWidth: 0 }}>{cellContent}</div>
    </TableCell>
  );
};

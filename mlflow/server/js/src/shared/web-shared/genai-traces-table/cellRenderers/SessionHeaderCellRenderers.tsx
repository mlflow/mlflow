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
  type ModelTraceInfoV3,
  isFeedbackAssessment,
} from '@databricks/web-shared/model-trace-explorer';

import { NullCell } from './NullCell';
import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { formatDateTime } from './rendererFunctions';
import { EvaluationsReviewAssessmentTag } from '../components/EvaluationsReviewAssessmentTag';
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
import { TracesTableColumnType, type TracesTableColumn } from '../types';
import { escapeCssSpecialCharacters } from '../utils/DisplayUtils';
import {
  convertFeedbackAssessmentToRunEvalAssessment,
  getTraceInfoInputs,
  getTraceInfoOutputs,
} from '../utils/TraceUtils';
import { compact } from 'lodash';
import { getUniqueValueCountsBySourceId } from '../utils/AggregationUtils';
import { TokenComponent } from './TokensCell';
import { SessionHeaderPassFailAggregatedCell } from './SessionHeaderPassFailAggregatedCell';

interface SessionHeaderCellProps {
  column: TracesTableColumn;
  sessionId: string;
  traces: ModelTraceInfoV3[];
  onChangeEvaluationId?: (evaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void;
  experimentId: string;
}

/**
 * Renders a cell in the session header row based on the column type.
 * Returns null for columns that don't have session-level representations.
 */
export const SessionHeaderCell: React.FC<SessionHeaderCellProps> = ({ column, sessionId, traces, experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
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
            <Typography.Text ellipsis>{sessionId}</Typography.Text>
          </Tag>
        </SessionIdLinkWrapper>
      </div>
    );
  } else if (column.id === INPUTS_COLUMN_ID && firstTrace) {
    // Request/Input column - get from first trace's request_preview
    const inputTitle = getTraceInfoInputs(firstTrace);
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
        <Typography.Text ellipsis>
          {formatDateTime(date, intl, {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
          })}
        </Typography.Text>
      </Tooltip>
    ) : (
      <NullCell />
    );
  } else if (column.id === USER_COLUMN_ID && firstTrace) {
    // User column - get from first trace's metadata or tags
    const value = firstTrace.trace_metadata?.[MLFLOW_TRACE_USER_KEY];
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

    cellContent = cellContent = (
      <TokenComponent
        inputTokens={totalInputTokens}
        outputTokens={totalOutputTokens}
        totalTokens={totalTokens}
        isComparing={false}
      />
    );
  } else if (
    column.type === TracesTableColumnType.ASSESSMENT &&
    column.assessmentInfo &&
    column.assessmentInfo?.isSessionLevelAssessment &&
    traces.length > 0
  ) {
    // Session-level assessment column - find the assessment with session metadata
    const assessmentInfo = column.assessmentInfo;
    const assessmentName = assessmentInfo.name;
    const allFeedback = traces.flatMap((trace) =>
      compact(
        trace.assessments?.filter(
          (a): a is FeedbackAssessment =>
            Boolean(a.metadata?.[ASSESSMENT_SESSION_METADATA_KEY]) && isFeedbackAssessment(a),
        ),
      ),
    );

    const entries = allFeedback.map(convertFeedbackAssessmentToRunEvalAssessment);
    const uniqueValueCounts = getUniqueValueCountsBySourceId(assessmentInfo, entries);

    cellContent = (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        {uniqueValueCounts.map((uniqueValueCount) => {
          const assessment = uniqueValueCount.latestAssessment;
          const count = uniqueValueCount.count;
          return (
            <EvaluationsReviewAssessmentTag
              key={`tag_${uniqueValueCount.latestAssessment.name}_${uniqueValueCount.value}`}
              showRationaleInTooltip
              disableJudgeTypeIcon
              hideAssessmentName
              assessment={assessment}
              isRootCauseAssessment={false}
              assessmentInfo={assessmentInfo}
              type="value"
              count={count}
            />
          );
        })}
      </div>
    );
  } else if (
    column.type === TracesTableColumnType.ASSESSMENT &&
    column.assessmentInfo &&
    !column.assessmentInfo?.isSessionLevelAssessment &&
    traces.length > 0
  ) {
    cellContent = <SessionHeaderPassFailAggregatedCell assessmentInfo={column.assessmentInfo} traces={traces} />;
  } else {
    cellContent = <NullCell />;
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

import { isNil } from 'lodash';
import React from 'react';

import {
  SpeechBubbleIcon,
  TableCell,
  Tag,
  Tooltip,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { NullCell } from './NullCell';
import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { formatDateTime } from './rendererFunctions';
import {
  INPUTS_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  SESSION_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  USER_COLUMN_ID,
} from '../hooks/useTableColumns';
import type { TracesTableColumn } from '../types';
import { escapeCssSpecialCharacters } from '../utils/DisplayUtils';

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
        <Typography.Text ellipsis>{value}</Typography.Text>
      </div>
    ) : (
      <NullCell />
    );
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

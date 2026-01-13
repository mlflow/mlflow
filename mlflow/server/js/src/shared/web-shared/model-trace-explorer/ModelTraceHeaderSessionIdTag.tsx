import { useCallback } from 'react';

import { SpeechBubbleIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getExperimentChatSessionPageRoute } from './MlflowUtils';
import { ModelTraceHeaderMetricSection } from './ModelTraceExplorerMetricSection';
import { Link, useLocation } from './RoutingUtils';
import { SELECTED_TRACE_ID_QUERY_PARAM } from './constants';

const ID_MAX_LENGTH = 10;

export const ModelTraceHeaderSessionIdTag = ({
  experimentId,
  sessionId,
  traceId,
  handleCopy,
}: {
  experimentId: string;
  sessionId: string;
  traceId?: string;
  handleCopy: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const location = useLocation();
  const truncatedSessionId = sessionId.length > ID_MAX_LENGTH ? `${sessionId.slice(0, ID_MAX_LENGTH)}...` : sessionId;
  const getTruncatedLabel = useCallback(
    (label: string) => (label.length > ID_MAX_LENGTH ? `${label.slice(0, ID_MAX_LENGTH)}...` : label),
    [],
  );

  const baseUrl = getExperimentChatSessionPageRoute(experimentId, sessionId);
  const sessionPageUrl = traceId
    ? `${baseUrl}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: traceId }).toString()}`
    : baseUrl;

  // If already on the session page, clicking the Session ID should copy it to clipboard
  // instead of navigating (which would be a no-op)
  const isOnSessionPage = location.pathname.includes(`/chat-sessions/${sessionId}`);

  if (isOnSessionPage) {
    return (
      <ModelTraceHeaderMetricSection
        label={<FormattedMessage defaultMessage="Session ID" description="Label for the session id section" />}
        icon={<SpeechBubbleIcon css={{ fontSize: 12, display: 'flex' }} />}
        value={sessionId}
        color="default"
        getTruncatedLabel={getTruncatedLabel}
        onCopy={handleCopy}
      />
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Text size="md" color="secondary">
        <FormattedMessage defaultMessage="Session ID" description="Label for the session id section" />
      </Typography.Text>
      <Tooltip
        componentId="mlflow.model-trace-explorer.session-id-tag"
        content={<FormattedMessage defaultMessage="View chat session" description="Tooltip for the session id tag" />}
      >
        <Link to={sessionPageUrl}>
          <Tag componentId="mlflow.model_trace_explorer.header_details.tag-session-id">
            <span css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: theme.spacing.xs }}>
              <SpeechBubbleIcon css={{ fontSize: 12 }} />
              <span>{truncatedSessionId}</span>
            </span>
          </Tag>
        </Link>
      </Tooltip>
    </div>
  );
};

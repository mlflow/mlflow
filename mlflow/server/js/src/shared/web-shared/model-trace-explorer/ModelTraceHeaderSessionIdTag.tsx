import { useCallback } from 'react';

import {
  NewWindowIcon,
  SpeechBubbleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { shouldEnableChatSessionsTab } from './FeatureUtils';
import { getExperimentChatSessionPageRoute } from './MlflowUtils';
import { ModelTraceHeaderMetricSection } from './ModelTraceExplorerMetricSection';
import { Link } from './RoutingUtils';

const ID_MAX_LENGTH = 10;

export const ModelTraceHeaderSessionIdTag = ({
  experimentId,
  sessionId,
  handleCopy,
}: {
  experimentId: string;
  sessionId: string;
  handleCopy: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const truncatedSessionId = sessionId.length > ID_MAX_LENGTH ? `${sessionId.slice(0, ID_MAX_LENGTH)}...` : sessionId;
  const getTruncatedLabel = useCallback(
    (label: string) => (label.length > ID_MAX_LENGTH ? `${label.slice(0, ID_MAX_LENGTH)}...` : label),
    [],
  );

  if (!shouldEnableChatSessionsTab()) {
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
        content={
          <Link
            to={getExperimentChatSessionPageRoute(experimentId, sessionId)}
            target="_blank"
            rel="noopener noreferrer"
          >
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                gap: theme.spacing.sm,
                color: theme.colors.actionPrimaryIcon,
              }}
            >
              <FormattedMessage defaultMessage="View chat session" description="Tooltip for the session id tag" />
              <NewWindowIcon />
            </div>
          </Link>
        }
      >
        <Link to={getExperimentChatSessionPageRoute(experimentId, sessionId)} target="_blank" rel="noopener noreferrer">
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

import {
  NewWindowIcon,
  SpeechBubbleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getExperimentChatSessionPageRoute } from './MlflowUtils';
import { Link } from './RoutingUtils';

const ID_MAX_LENGTH = 10;

export const ModelTraceHeaderSessionIdTag = ({
  experimentId,
  sessionId,
}: {
  experimentId: string;
  sessionId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const truncatedSessionId = sessionId.length > ID_MAX_LENGTH ? `${sessionId.slice(0, ID_MAX_LENGTH)}...` : sessionId;

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

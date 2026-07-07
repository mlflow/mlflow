import { ClockIcon, Tag, TokenIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ExperimentSingleChatMetrics } from './useExperimentSingleChatMetrics';

import { FormattedMessage } from '@databricks/i18n';

export const ExperimentSingleChatSessionMetrics = ({
  chatSessionMetrics,
}: {
  chatSessionMetrics: ExperimentSingleChatMetrics;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        borderBottom: `1px solid ${theme.colors.grey100}`,
        paddingBottom: theme.spacing.sm,
        display: 'flex',
        gap: theme.spacing.sm,
      }}
    >
      {chatSessionMetrics.sessionTokens.total_tokens && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Tokens"
              description="Label for the total token count metric in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.tokens-tag" icon={<TokenIcon />}>
            {chatSessionMetrics.sessionTokens.total_tokens}
          </Tag>
        </div>
      )}
      {chatSessionMetrics.sessionLatency && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Latency"
              description="Label for the total latency metric in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.latency-tag" icon={<ClockIcon />}>
            <FormattedMessage
              defaultMessage="{latency} sec"
              description="Display format for latency value in seconds in chat session metrics"
              values={{ latency: chatSessionMetrics.sessionLatency?.toFixed(4) }}
            />
          </Tag>
        </div>
      )}
      {chatSessionMetrics.goal && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Goal"
              description="Label for the simulation goal metadata in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.goal-tag">
            <Tooltip
              componentId="mlflow.experiment.chat-session.metrics.goal-tooltip"
              content={chatSessionMetrics.goal}
            >
              <span
                css={{
                  maxWidth: 200,
                  display: 'inline-block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  verticalAlign: 'bottom',
                }}
              >
                {chatSessionMetrics.goal}
              </span>
            </Tooltip>
          </Tag>
        </div>
      )}
      {chatSessionMetrics.persona && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Persona"
              description="Label for the simulation persona metadata in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.persona-tag">
            <Tooltip
              componentId="mlflow.experiment.chat-session.metrics.persona-tooltip"
              content={chatSessionMetrics.persona}
            >
              <span
                css={{
                  maxWidth: 200,
                  display: 'inline-block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  verticalAlign: 'bottom',
                }}
              >
                {chatSessionMetrics.persona}
              </span>
            </Tooltip>
          </Tag>
        </div>
      )}
    </div>
  );
};

import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ExperimentSingleChatMetrics } from './useExperimentSingleChatMetrics';
import { FormattedMessage } from 'react-intl';

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
        paddingLeft: theme.spacing.md,
        paddingRight: theme.spacing.md,
        display: 'flex',
        gap: theme.spacing.sm,
      }}
    >
      {chatSessionMetrics.sessionTokens.total_tokens > 0 && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
            alignItems: 'center',
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Tokens:"
              description="Label for the total token count metric in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.tokens-tag">
            {chatSessionMetrics.sessionTokens.total_tokens}
          </Tag>
        </div>
      )}
      {chatSessionMetrics.sessionLatency && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
            alignItems: 'center',
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Latency:"
              description="Label for the total latency metric in chat session metrics"
            />
          </Typography.Text>
          <Tag componentId="mlflow.experiment.chat-session.metrics.latency-tag">
            <FormattedMessage
              defaultMessage="{latency} sec"
              description="Display format for latency value in seconds in chat session metrics"
              values={{ latency: chatSessionMetrics.sessionLatency?.toFixed(4) }}
            />
          </Tag>
        </div>
      )}
    </div>
  );
};

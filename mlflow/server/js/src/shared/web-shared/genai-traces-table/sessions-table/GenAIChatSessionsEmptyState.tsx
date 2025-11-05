import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const GenAIChatSessionsEmptyState = () => {
  return (
    <div css={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="Group traces from the same chat session together"
          description="Empty state title for the chat sessions table"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
        <FormattedMessage
          defaultMessage="MLflow allows"
          description="Empty state description for the chat sessions table"
        />
      </Typography.Paragraph>
    </div>
  );
};

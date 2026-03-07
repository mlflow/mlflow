import { SpeechBubbleIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CopyActionButton } from '@databricks/web-shared/copy';
import { TracesV3DateSelector } from './TracesV3DateSelector';
import { FormattedMessage } from '@databricks/i18n';

export const TracesV3Toolbar = ({
  viewState,
  sessionId,
  className,
}: {
  viewState: string;
  sessionId?: string;
  className?: string;
}) => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        paddingBottom: `${theme.spacing.sm}px`,
      }}
      className={className}
    >
      {/**
       * in the single chat session view, the date sector is irrelevant as we
       * want to show all traces in the session regardless date.
       * additionally, we want to show the session ID in the title bar so
       * the user knows which session they are viewing.
       */}
      {!(viewState === 'single-chat-session') && <TracesV3DateSelector />}
      {viewState === 'single-chat-session' && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Tag
            color="default"
            componentId="mlflow.chat-sessions.session-header-label"
            css={{ padding: theme.spacing.xs + 2, margin: 0 }}
          >
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <SpeechBubbleIcon />
              <Typography.Text ellipsis bold>
                <FormattedMessage defaultMessage="Session" description="Label preceding a chat session ID" />
              </Typography.Text>
            </span>
          </Tag>
          <Typography.Title level={3} withoutMargins>
            {sessionId}
          </Typography.Title>
          {sessionId && (
            <CopyActionButton
              copyText={sessionId}
              componentId="mlflow.chat-sessions.copy-session-id"
              buttonProps={{ icon: undefined }}
            />
          )}
        </div>
      )}
    </div>
  );
};

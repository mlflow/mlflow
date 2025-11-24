import { CopyIcon, SpeechBubbleIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { TracesV3DateSelector } from './TracesV3DateSelector';
import { FormattedMessage } from '@databricks/i18n';
import { useCallback, useState } from 'react';

export const TracesV3Toolbar = ({ viewState, sessionId }: { viewState: string; sessionId?: string }) => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { theme } = useDesignSystemTheme();
  const [showCopyTooltip, setShowCopyTooltip] = useState(false);

  const handleCopySessionId = useCallback(() => {
    if (sessionId) {
      navigator.clipboard.writeText(sessionId);
      setShowCopyTooltip(true);
      setTimeout(() => setShowCopyTooltip(false), 2000);
    }
  }, [sessionId]);

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        borderBottom: `1px solid ${theme.colors.grey100}`,
        paddingBottom: `${theme.spacing.sm}px`,
      }}
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
          <Tooltip
            componentId="mlflow.chat-sessions.copy-session-id"
            content={
              showCopyTooltip ? (
                <FormattedMessage defaultMessage="Copied!" description="Tooltip after copying session ID" />
              ) : (
                <FormattedMessage defaultMessage="Copy session ID" description="Tooltip for copy session ID button" />
              )
            }
            open={showCopyTooltip ? true : undefined}
          >
            <CopyIcon
              onClick={handleCopySessionId}
              css={{
                cursor: 'pointer',
                color: theme.colors.textSecondary,
                fontSize: 16,
                '&:hover': {
                  color: theme.colors.textPrimary,
                },
              }}
            />
          </Tooltip>
        </div>
      )}
    </div>
  );
};

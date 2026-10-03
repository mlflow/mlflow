import { isNil } from 'lodash';
import { useMemo, useRef, useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerChatMessage } from './ModelTraceExplorerChatMessage';
import type { ModelTraceChatMessage } from '../ModelTrace.types';

export function ModelTraceExplorerConversation({
  messages,
  maxVisibleMessages,
}: {
  messages: ModelTraceChatMessage[] | null;
  maxVisibleMessages?: number;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const [messagesExpanded, setMessagesExpanded] = useState(false);
  const previousMessagesRef = useRef(messages);
  if (previousMessagesRef.current !== messages) {
    previousMessagesRef.current = messages;
    if (messagesExpanded) {
      setMessagesExpanded(false);
    }
  }
  const toolCallNameById = useMemo(() => {
    const toolCallNames = new Map<string, string>();

    for (const message of messages ?? []) {
      for (const toolCall of message.tool_calls ?? []) {
        toolCallNames.set(toolCall.id, toolCall.function.name);
      }
    }

    return toolCallNames;
  }, [messages]);

  if (isNil(messages)) {
    return null;
  }

  const visibleMessageLimit = maxVisibleMessages ?? messages.length;
  const shouldTruncateMessages = messages.length > visibleMessageLimit;
  const visibleMessages = messagesExpanded || !shouldTruncateMessages ? messages : messages.slice(-visibleMessageLimit);
  const hiddenMessageCount = messages.length - visibleMessages.length;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      {shouldTruncateMessages && (
        <Typography.Link
          css={{ alignSelf: 'flex-start', marginLeft: theme.spacing.xs }}
          componentId="shared.model-trace-explorer.conversation-toggle"
          onClick={() => setMessagesExpanded((expanded) => !expanded)}
        >
          {messagesExpanded ? (
            <FormattedMessage
              defaultMessage="Show less"
              description="Button label to collapse conversation messages in model trace explorer"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Show {hiddenMessageCount} more"
              description="Button label to expand and show hidden conversation messages in model trace explorer"
              values={{ hiddenMessageCount }}
            />
          )}
        </Typography.Link>
      )}
      {visibleMessages.map((message, index) => (
        <ModelTraceExplorerChatMessage
          css={{
            // Render each message as a left-aligned chat bubble.
            maxWidth: '90%',
            alignSelf: 'flex-start',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
          }}
          key={index}
          message={message}
          toolCallName={message.tool_call_id ? toolCallNameById.get(message.tool_call_id) : undefined}
        />
      ))}
    </div>
  );
}

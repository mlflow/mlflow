import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerChatMessage } from './ModelTraceExplorerChatMessage';
import type { ModelTraceChatMessage } from '../ModelTrace.types';

export function ModelTraceExplorerConversation({
  messages,
}: {
  messages: ModelTraceChatMessage[] | null;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
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

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      {messages.map((message, index) => (
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

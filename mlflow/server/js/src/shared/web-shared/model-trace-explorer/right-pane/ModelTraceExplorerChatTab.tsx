import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerChatTool } from './ModelTraceExplorerChatTool';
import { ModelTraceExplorerConversation } from './ModelTraceExplorerConversation';
import type { ModelTraceChatMessage, ModelTraceChatTool } from '../ModelTrace.types';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';

export function ModelTraceExplorerChatTab({
  chatMessages,
  chatTools,
}: {
  chatMessages: ModelTraceChatMessage[];
  chatTools?: ModelTraceChatTool[];
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        overflowY: 'auto',
        padding: theme.spacing.md,
      }}
      data-testid="model-trace-explorer-chat-tab"
    >
      {chatTools && (
        <ModelTraceExplorerCollapsibleSection
          css={{ marginBottom: theme.spacing.sm }}
          title={
            <FormattedMessage
              defaultMessage="Tools"
              description="Section header in the chat tab that displays all tools that were available for the chat model to call during execution"
            />
          }
          sectionKey="messages"
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {chatTools.map((tool) => (
              <ModelTraceExplorerChatTool key={tool.function.name} tool={tool} />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}

      <ModelTraceExplorerCollapsibleSection
        title={
          <FormattedMessage
            defaultMessage="Messages"
            description="Section header in the chat tab that displays the message history between the user and the chat model"
          />
        }
        sectionKey="messages"
      >
        <ModelTraceExplorerConversation messages={chatMessages} />
      </ModelTraceExplorerCollapsibleSection>
    </div>
  );
}

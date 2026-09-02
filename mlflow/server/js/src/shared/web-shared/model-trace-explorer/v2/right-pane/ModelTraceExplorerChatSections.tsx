import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerChatTool } from './ModelTraceExplorerChatTool';
import { ModelTraceExplorerConversation } from './ModelTraceExplorerConversation';
import type { ModelTraceChatMessage, ModelTraceChatTool } from '../ModelTrace.types';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';

export function ModelTraceExplorerChatSections({
  messages,
  tools,
  className,
}: {
  messages?: ModelTraceChatMessage[] | null;
  tools?: ModelTraceChatTool[];
  className?: string;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const visibleTools = tools ?? [];
  const visibleMessages = messages ?? [];
  const hasTools = visibleTools.length > 0;
  const hasMessages = visibleMessages.length > 0;

  if (!hasTools && !hasMessages) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }} className={className}>
      {hasTools && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          css={{ marginBottom: hasMessages ? theme.spacing.sm : 0 }}
          headerPadding={`${theme.spacing.xs}px 0`}
          contentPadding={`${theme.spacing.xs}px 0 ${theme.spacing.xs}px ${theme.spacing.sm + theme.spacing.xs}px`}
          initialExpanded={false}
          title={
            <FormattedMessage
              defaultMessage="Tools ({count})"
              description="Section header in the model trace explorer inputs section that displays all tools available for the chat model to call during execution"
              values={{ count: visibleTools.length }}
            />
          }
          sectionKey="tools"
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {visibleTools.map((tool) => (
              <ModelTraceExplorerChatTool key={tool.function.name} tool={tool} />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}

      {hasMessages && <ModelTraceExplorerConversation messages={visibleMessages} />}
    </div>
  );
}

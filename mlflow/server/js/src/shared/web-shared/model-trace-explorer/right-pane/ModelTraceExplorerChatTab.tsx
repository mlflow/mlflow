import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerChatTool } from './ModelTraceExplorerChatTool';
import { ModelTraceExplorerConversation } from './ModelTraceExplorerConversation';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { OpenInPlaygroundButton } from '../OpenInPlaygroundButton';
import { SpanModelCostBadge } from './SpanModelCostBadge';

export function ModelTraceExplorerChatTab({ activeSpan }: { activeSpan: ModelTraceSpanNode }) {
  const { theme } = useDesignSystemTheme();
  const { chatMessages, chatTools } = activeSpan;

  return (
    <div
      css={{
        overflowY: 'auto',
      }}
      data-testid="model-trace-explorer-chat-tab"
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          marginBlock: theme.spacing.sm,
          marginLeft: theme.spacing.sm,
          marginRight: theme.spacing.sm,
        }}
      >
        <SpanModelCostBadge activeSpan={activeSpan} />
        <OpenInPlaygroundButton traceId={activeSpan.traceId} spanId={String(activeSpan.key)} size="small" />
      </div>
      {chatTools && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
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
        withBorder
      >
        <ModelTraceExplorerConversation messages={chatMessages ?? []} />
      </ModelTraceExplorerCollapsibleSection>
    </div>
  );
}

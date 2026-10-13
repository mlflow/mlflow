import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceChatTool } from '../ModelTrace.types';
import { ModelTraceExplorerChatTool } from '../right-pane/ModelTraceExplorerChatTool';

export const ModelTraceExplorerChatToolsRenderer = ({
  title,
  tools,
}: {
  title: string;
  tools: ModelTraceChatTool[];
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: 0,
        border: 'none',
        borderRadius: 0,
      }}
    >
      {title && (
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography.Text css={{ marginLeft: theme.spacing.xs }} bold>
            {title}
          </Typography.Text>
        </div>
      )}
      {tools.map((tool) => (
        <ModelTraceExplorerChatTool key={tool.function.name} tool={tool} />
      ))}
    </div>
  );
};

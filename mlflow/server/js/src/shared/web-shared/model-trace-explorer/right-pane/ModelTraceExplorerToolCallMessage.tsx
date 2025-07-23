import { FunctionIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceToolCall } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from '../ModelTraceExplorerCodeSnippetBody';

export function ModelTraceExplorerToolCallMessage({ toolCall }: { toolCall: ModelTraceToolCall }) {
  const { theme } = useDesignSystemTheme();

  return (
    <div key={toolCall.id} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <Typography.Text
        color="secondary"
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          padding: `0px ${theme.spacing.sm + theme.spacing.xs}px`,
        }}
      >
        <FormattedMessage
          defaultMessage="called {functionName} in {toolCallId}"
          description="A message that shows the tool calls that an AI assistant made. The full message reads (for example): 'Assistant called get_weather in id_123'."
          values={{
            functionName: (
              <Tag
                color="purple"
                componentId="shared.model-trace-explorer.function-name-tag"
                css={{ margin: `0px ${theme.spacing.xs}px` }}
              >
                <FunctionIcon />
                <Typography.Text css={{ whiteSpace: 'nowrap', marginLeft: theme.spacing.xs }}>
                  {toolCall.function.name}
                </Typography.Text>
              </Tag>
            ),
            toolCallId: (
              <Tooltip componentId="test" content={toolCall.id}>
                <div css={{ display: 'inline-flex', flexShrink: 1, overflow: 'hidden', marginLeft: theme.spacing.xs }}>
                  <Typography.Text
                    css={{
                      textOverflow: 'ellipsis',
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                    }}
                    code
                    color="secondary"
                  >
                    {toolCall.id}
                  </Typography.Text>
                </div>
              </Tooltip>
            ),
          }}
        />
      </Typography.Text>
      <ModelTraceExplorerCodeSnippetBody data={toolCall.function.arguments} />
    </div>
  );
}

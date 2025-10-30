import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTrace } from '../ModelTrace.types';
import { parseModelTraceToTree, createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { ModelTraceExplorerChatMessage } from '../right-pane/ModelTraceExplorerChatMessage';

export const SingleChatTurnMessages = ({ trace }: { trace: ModelTrace }) => {
  const { theme } = useDesignSystemTheme();

  const spans = trace.data?.spans;
  if (!spans) {
    return null;
  }

  const rootSpan = parseModelTraceToTree(trace);
  if (!rootSpan) {
    return null;
  }

  // if they exist, slice from the last user message
  const chatMessages = rootSpan.chatMessages;
  const displayedMessages =
    chatMessages?.slice(chatMessages?.findLastIndex((message) => message.role === 'user')) ?? [];

  if (displayedMessages.length !== 0) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {displayedMessages?.map((message, index) => (
          <ModelTraceExplorerChatMessage
            key={index}
            message={message}
            css={{ maxWidth: '60%', alignSelf: message.role === 'user' ? 'flex-start' : 'flex-end' }}
          />
        ))}
      </div>
    );
  }

  // take the first input and output. maybe in the future this can be
  // expandable, but for now the user can simply click on the trace to
  // view the full input and output.
  const inputList = createListFromObject(rootSpan.inputs);
  const outputList = createListFromObject(rootSpan.outputs);

  // try to get the first nonnull if possible
  const input = inputList.find((item) => item.value !== null) ?? inputList[0];
  const output = outputList.find((item) => item.value !== null) ?? outputList[0];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          width: '80%',
          alignSelf: 'flex-start',
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <ModelTraceExplorerFieldRenderer
          title={input?.key ?? 'input'}
          data={input?.value ?? 'null'}
          renderMode="text"
        />
      </div>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          width: '80%',
          alignSelf: 'flex-end',
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <ModelTraceExplorerFieldRenderer
          title={output?.key ?? 'output'}
          data={output?.value ?? 'null'}
          renderMode="text"
        />
      </div>
    </div>
  );
};

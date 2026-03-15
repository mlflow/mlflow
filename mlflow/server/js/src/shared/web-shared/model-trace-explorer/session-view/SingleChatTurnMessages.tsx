import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace, ModelTraceChatMessage } from '../ModelTrace.types';
import { parseModelTraceToTree, createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerChatMessage } from '../right-pane/ModelTraceExplorerChatMessage';
import { ModelTraceExplorerSummarySection } from '../summary-view/ModelTraceExplorerSummarySection';

const QUERY_FIELD_NAMES = ['query', 'input', 'message', 'question', 'prompt', 'content'];

/**
 * When standard chat message parsing fails, try to extract a simple
 * user/assistant pair from raw inputs/outputs. This handles frameworks
 * like LangGraph where inputs contain `{ query: "...", thread: {...} }`
 * and outputs are a plain string.
 */
export function extractSimpleChatMessages(
  inputs: Record<string, unknown> | string | undefined | null,
  outputs: Record<string, unknown> | string | undefined | null,
): ModelTraceChatMessage[] | null {
  // The assistant response must be a plain string
  if (typeof outputs !== 'string' || !outputs) {
    return null;
  }

  let userContent: string | undefined;

  if (typeof inputs === 'string' && inputs) {
    userContent = inputs;
  } else if (inputs && typeof inputs === 'object') {
    const hasMessages =
      'messages' in inputs ||
      Object.values(inputs).some((v) => v && typeof v === 'object' && !Array.isArray(v) && 'messages' in v);
    if (!hasMessages) {
      return null;
    }
    for (const field of QUERY_FIELD_NAMES) {
      if (typeof inputs[field] === 'string' && inputs[field]) {
        userContent = inputs[field];
        break;
      }
    }
  }

  if (!userContent) {
    return null;
  }

  return [
    { role: 'user', content: userContent },
    { role: 'assistant', content: outputs },
  ];
}

export const PREFERRED_INPUT_KEYS = ['messages', 'input', 'inputs'];
export const PREFERRED_OUTPUT_KEYS = ['response', 'output', 'outputs', 'generations'];

export const rankByKeyImportance = (preferredKeys: string[]) => {
  return (a: { key: string }, b: { key: string }): number => {
    const aIndex = preferredKeys.indexOf(a.key.toLowerCase());
    const bIndex = preferredKeys.indexOf(b.key.toLowerCase());
    // Both are preferred: sort by preference order
    if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
    // Only one is preferred: it comes first
    if (aIndex !== -1) return -1;
    if (bIndex !== -1) return 1;
    // Neither is preferred: preserve original order
    return 0;
  };
};

const rankInputByImportance = rankByKeyImportance(PREFERRED_INPUT_KEYS);
const rankOutputByImportance = rankByKeyImportance(PREFERRED_OUTPUT_KEYS);

export const SingleChatTurnMessages = ({ trace }: { trace: ModelTrace }) => {
  const { theme } = useDesignSystemTheme();

  const rootSpan = useMemo(() => (trace.data?.spans ? parseModelTraceToTree(trace) : null), [trace]);

  if (!rootSpan) {
    return null;
  }

  // For the session view, show only the last user message and the final assistant
  // response (skip tool calls, tool results, and intermediate assistant messages).
  const chatMessages = rootSpan.chatMessages;
  const lastUserIdx = chatMessages?.findLastIndex((message) => message.role === 'user') ?? -1;
  const lastAssistantMessages = chatMessages
    ? chatMessages.slice(lastUserIdx + 1).filter((m) => m.role === 'assistant' && m.content && !m.tool_calls?.length)
    : [];
  const displayedMessages = [
    ...(lastUserIdx >= 0 && chatMessages ? [chatMessages[lastUserIdx]] : []),
    ...(lastAssistantMessages.length > 0 ? [lastAssistantMessages[lastAssistantMessages.length - 1]] : []),
  ];

  if (displayedMessages.length !== 0) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {displayedMessages?.map((message, index) => (
          <ModelTraceExplorerChatMessage
            key={index}
            message={message}
            css={{
              maxWidth: '80%',
              alignSelf: message.role === 'user' ? 'flex-start' : 'flex-end',
              borderWidth: 2,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          />
        ))}
      </div>
    );
  }

  // Fallback: try to extract a simple user/assistant pair from raw inputs/outputs
  const simpleChatMessages = extractSimpleChatMessages(rootSpan.inputs, rootSpan.outputs);
  if (simpleChatMessages) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {simpleChatMessages.map((message, index) => (
          <ModelTraceExplorerChatMessage
            key={index}
            message={message}
            css={{
              maxWidth: '80%',
              alignSelf: message.role === 'user' ? 'flex-start' : 'flex-end',
              borderWidth: 2,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          />
        ))}
      </div>
    );
  }

  // Sort by importance then reverse — the component expands upwards,
  // so the last item in the array is the one visible above the fold.
  const inputList = createListFromObject(rootSpan.inputs)
    .filter((item) => item.value !== 'null')
    .sort(rankInputByImportance)
    .reverse();
  const outputList = createListFromObject(rootSpan.outputs)
    .filter((item) => item.value !== 'null')
    .sort(rankOutputByImportance)
    .reverse();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          width: '80%',
          alignSelf: 'flex-start',
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <ModelTraceExplorerSummarySection
          title={
            <FormattedMessage
              defaultMessage="Inputs"
              description="Section title for the inputs of a single chat turn"
            />
          }
          data={inputList}
          renderMode="default"
          sectionKey="summary-inputs"
          maxVisibleItems={1}
          maxVisibleChatMessages={1}
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
        <ModelTraceExplorerSummarySection
          title={
            <FormattedMessage
              defaultMessage="Outputs"
              description="Section title for the outputs of a single chat turn"
            />
          }
          data={outputList}
          renderMode="default"
          sectionKey="summary-outputs"
          maxVisibleItems={1}
          maxVisibleChatMessages={1}
        />
      </div>
    </div>
  );
};

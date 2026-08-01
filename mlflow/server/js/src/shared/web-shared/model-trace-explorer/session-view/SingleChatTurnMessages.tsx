import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace, ModelTraceChatMessage, ModelTraceSpanNode } from '../ModelTrace.types';
import { parseModelTraceToTree, createListFromObject } from '../ModelTraceExplorer.utils';
import { normalizeLangchainChatInput } from '../chat-utils/langchain';
import { ModelTraceExplorerChatMessage } from '../right-pane/ModelTraceExplorerChatMessage';
import { ModelTraceExplorerSummarySection } from '../summary-view/ModelTraceExplorerSummarySection';

/**
 * When standard chat message parsing fails, try to extract a simple
 * user/assistant pair from raw inputs/outputs. This handles frameworks
 * like LangGraph where both inputs and outputs contain a `messages` key
 * with LangChain-style message objects (e.g. `{ messages: [{ type: "human", content: "..." }] }`).
 */
export function extractSimpleChatMessages(
  inputs: Record<string, unknown> | string | undefined | null,
  outputs: Record<string, unknown> | string | undefined | null,
): ModelTraceChatMessage[] | null {
  // Try to parse LangGraph/LangChain messages format from inputs
  const inputMessages = inputs ? normalizeLangchainChatInput(inputs) : null;
  if (inputMessages && inputMessages.length > 0) {
    const lastUser = inputMessages.findLast((m) => m.role === 'user');
    if (!lastUser) {
      return null;
    }

    // Try to get the assistant response from outputs messages
    const outputMessages = outputs ? normalizeLangchainChatInput(outputs) : null;
    if (outputMessages && outputMessages.length > 0) {
      const lastAssistant = outputMessages.findLast(
        (m) => m.role === 'assistant' && m.content && !m.tool_calls?.length,
      );
      if (lastAssistant) {
        return [lastUser, lastAssistant];
      }
    }

    // Fall back to string output
    if (typeof outputs === 'string' && outputs) {
      return [lastUser, { role: 'assistant', content: outputs }];
    }

    return null;
  }

  // Simple string input/output fallback
  if (typeof inputs === 'string' && inputs && typeof outputs === 'string' && outputs) {
    return [
      { role: 'user', content: inputs },
      { role: 'assistant', content: outputs },
    ];
  }

  return null;
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

type ChatTurnContent =
  | { type: 'messages'; messages: ModelTraceChatMessage[] }
  | { type: 'raw'; inputList: { key: string; value: string }[]; outputList: { key: string; value: string }[] };

/**
 * For the session view, show only the last user message and the final assistant
 * response (skip tool calls, tool results, and intermediate assistant messages).
 */
export const getDisplayedChatMessages = (
  chatMessages: ModelTraceChatMessage[] | undefined,
): ModelTraceChatMessage[] => {
  const lastUserIdx = chatMessages?.findLastIndex((message) => message.role === 'user') ?? -1;
  const lastAssistantMessages = chatMessages
    ? chatMessages.slice(lastUserIdx + 1).filter((m) => m.role === 'assistant' && m.content && !m.tool_calls?.length)
    : [];

  return [
    ...(lastUserIdx >= 0 && chatMessages ? [chatMessages[lastUserIdx]] : []),
    ...(lastAssistantMessages.length > 0 ? [lastAssistantMessages[lastAssistantMessages.length - 1]] : []),
  ];
};

const getRawContent = (span: ModelTraceSpanNode): Extract<ChatTurnContent, { type: 'raw' }> => ({
  type: 'raw',
  // Sort by importance then reverse — the component expands upwards,
  // so the last item in the array is the one visible above the fold.
  inputList: createListFromObject(span.inputs)
    .filter((item) => item.value !== 'null')
    .sort(rankInputByImportance)
    .reverse(),
  outputList: createListFromObject(span.outputs)
    .filter((item) => item.value !== 'null')
    .sort(rankOutputByImportance)
    .reverse(),
});

/** The content a single span can contribute to a turn, or null if it has none. */
export const getChatTurnContent = (span: ModelTraceSpanNode): ChatTurnContent | null => {
  const displayedMessages = getDisplayedChatMessages(span.chatMessages);
  if (displayedMessages.length > 0) {
    return { type: 'messages', messages: displayedMessages };
  }

  const simpleChatMessages = extractSimpleChatMessages(span.inputs, span.outputs);
  if (simpleChatMessages) {
    return { type: 'messages', messages: simpleChatMessages };
  }

  const rawContent = getRawContent(span);
  if (rawContent.inputList.length > 0 || rawContent.outputList.length > 0) {
    return rawContent;
  }

  return null;
};

/**
 * Resolve the content for a turn, preferring the root span.
 *
 * OpenTelemetry GenAI instrumentation puts the messages on the LLM span, which is
 * a child of an outer agent span, so a root-only lookup renders an empty turn. Walk
 * the tree breadth-first for the shallowest span that has something to show.
 */
export const findChatTurnContent = (rootSpan: ModelTraceSpanNode): ChatTurnContent => {
  // Index-based cursor rather than shift(), which re-indexes the queue each step.
  const queue: ModelTraceSpanNode[] = [rootSpan];

  for (let cursor = 0; cursor < queue.length; cursor++) {
    const span = queue[cursor];

    const content = getChatTurnContent(span);
    if (content) {
      return content;
    }

    queue.push(...(span.children ?? []));
  }

  // Nothing anywhere in the trace: keep rendering the root's empty sections.
  return getRawContent(rootSpan);
};

export const SingleChatTurnMessages = ({ trace }: { trace: ModelTrace }) => {
  const { theme } = useDesignSystemTheme();

  const rootSpan = useMemo(() => (trace.data?.spans ? parseModelTraceToTree(trace) : null), [trace]);

  if (!rootSpan) {
    return null;
  }

  const turnContent = findChatTurnContent(rootSpan);

  if (turnContent.type === 'messages') {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {turnContent.messages.map((message, index) => (
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

  const { inputList, outputList } = turnContent;

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
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            '& > div:first-of-type': {
              borderTopLeftRadius: theme.borders.borderRadiusMd,
              borderTopRightRadius: theme.borders.borderRadiusMd,
              borderTop: 'none',
            },
          }}
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
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <ModelTraceExplorerSummarySection
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            '& > div:first-of-type': {
              borderTopLeftRadius: theme.borders.borderRadiusMd,
              borderTopRightRadius: theme.borders.borderRadiusMd,
              borderTop: 'none',
            },
          }}
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

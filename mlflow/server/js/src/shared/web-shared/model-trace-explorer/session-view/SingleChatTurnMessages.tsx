import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace, ModelTraceChatMessage } from '../ModelTrace.types';
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

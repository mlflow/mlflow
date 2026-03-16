import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTrace } from '../ModelTrace.types';
import { parseModelTraceToTree, createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerChatMessage } from '../right-pane/ModelTraceExplorerChatMessage';
import { ModelTraceExplorerSummarySection } from '../summary-view/ModelTraceExplorerSummarySection';

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

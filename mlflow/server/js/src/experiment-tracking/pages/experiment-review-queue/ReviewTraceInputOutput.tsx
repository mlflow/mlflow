import { useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';
import { ModelTraceExplorerChatMessage, normalizeConversation } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTraceChatMessage } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage } from 'react-intl';

// Conversation formats to try, matching the adjacent review surface
// (GenAiEvaluationTracesReview.utils.tsx). `normalizeConversation` with no format
// only covers OpenAI/LangChain/Gemini, so pass each explicitly to also catch
// Anthropic-shaped payloads.
const COMMON_MESSAGE_FORMATS = ['openai', 'langchain', 'anthropic'];

// Parse a preview into JSON; return the raw string if it isn't valid JSON.
const parsePreview = (raw: string): unknown => {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
};

const toMessages = (parsed: unknown): ModelTraceChatMessage[] | null => {
  for (const format of COMMON_MESSAGE_FORMATS) {
    const messages = normalizeConversation(parsed, format);
    if (messages && messages.length > 0) {
      return messages;
    }
  }
  return null;
};

type Section =
  | { kind: 'chat'; messages: ModelTraceChatMessage[] }
  | { kind: 'markdown'; text: string }
  | { kind: 'json'; text: string };

/**
 * Decide how to render a preview, mirroring the Review App's preference order: a
 * parsed conversation first, then readable markdown for plain text, then JSON.
 * Exported for testing.
 */
export const deriveSection = (raw: string): Section => {
  const parsed = parsePreview(raw);

  if (typeof parsed === 'object' && parsed !== null) {
    const messages = toMessages(parsed);
    return messages ? { kind: 'chat', messages } : { kind: 'json', text: JSON.stringify(parsed, null, 2) };
  }

  // A string that failed to parse but looks like JSON (e.g. a payload truncated
  // by the server's preview cap) is shown as a JSON block, not run through the
  // markdown renderer where stray `*`/`#`/backticks would mangle it.
  if (typeof parsed === 'string') {
    return /^\s*[[{]/.test(parsed) ? { kind: 'json', text: parsed } : { kind: 'markdown', text: parsed };
  }

  // Scalars (number / boolean / null) — show their literal form.
  return { kind: 'json', text: raw };
};

/**
 * Renders a trace's input and output for the review focus view, mirroring the
 * Databricks Review App's `AgentInputOutput`: an Input card and an Output card,
 * each rendered for readability rather than as raw JSON. A payload that parses as
 * a conversation renders as chat messages via `ModelTraceExplorerChatMessage`
 * (role + markdown); a plain-text payload renders as markdown (the same engine
 * those messages use); anything else falls back to pretty-printed JSON. The full
 * trace stays available via the caller's full-trace drawer.
 */
export const ReviewTraceInputOutput = ({
  requestPreview,
  responsePreview,
}: {
  requestPreview?: string;
  responsePreview?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  const input = useMemo(() => (requestPreview ? deriveSection(requestPreview) : null), [requestPreview]);
  const output = useMemo(() => (responsePreview ? deriveSection(responsePreview) : null), [responsePreview]);

  const renderCard = (key: string, label: React.ReactNode, section: Section) => (
    <div
      key={key}
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Text bold color="secondary">
        {label}
      </Typography.Text>
      {section.kind === 'chat' ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {section.messages.map((message, i) => (
            // eslint-disable-next-line react/no-array-index-key
            <ModelTraceExplorerChatMessage key={i} message={message} />
          ))}
        </div>
      ) : section.kind === 'markdown' ? (
        <GenAIMarkdownRenderer>{section.text}</GenAIMarkdownRenderer>
      ) : (
        <pre
          css={{
            margin: 0,
            padding: theme.spacing.sm,
            backgroundColor: theme.colors.backgroundPrimary,
            borderRadius: theme.borders.borderRadiusSm,
            fontFamily: 'monospace',
            fontSize: theme.typography.fontSizeSm,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {section.text}
        </pre>
      )}
    </div>
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {input &&
        renderCard(
          'input',
          <FormattedMessage defaultMessage="Input" description="Review focused view: trace input label" />,
          input,
        )}
      {output &&
        renderCard(
          'output',
          <FormattedMessage defaultMessage="Output" description="Review focused view: trace output label" />,
          output,
        )}
    </div>
  );
};

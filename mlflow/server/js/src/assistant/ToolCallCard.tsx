/**
 * Renders a single tool call the assistant made, inline in the transcript: a
 * collapsible card whose header shows the tool name, a status badge, and a one-line
 * input summary, and whose expanded body shows the full input and (truncated) output.
 */
import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import {
  CheckCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  Spinner,
  Typography,
  WrenchSparkleIcon,
  XCircleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { ToolCallStatus, type AssistantPart } from './types';
import { GenAIMarkdownRenderer } from '../shared/web-shared/genai-markdown-renderer';

export type ToolCallPart = Extract<AssistantPart, { type: 'toolCall' }>;

// Tool output can be huge (e.g. a full trace); cap what we render in the card.
const MAX_OUTPUT_CHARS = 4000;

const truncate = (text: string): string =>
  text.length > MAX_OUTPUT_CHARS
    ? `${text.slice(0, MAX_OUTPUT_CHARS)}\n… (truncated, ${text.length - MAX_OUTPUT_CHARS} more chars)`
    : text;

// One-line summary of a tool call's input for the header. Generic across tools: joins the
// input's string values in order; falls back to compact JSON when there are none.
const toolInputSummary = (part: ToolCallPart): string => {
  const input = part.input ?? {};
  const strings = Object.values(input).filter((v): v is string => typeof v === 'string' && v.length > 0);
  if (strings.length > 0) return strings.join(' · ');
  const json = JSON.stringify(input);
  return json === '{}' ? '' : json;
};

// Fixed-size badge box so done/error/running all occupy identical space and
// keep every tool-call row's columns aligned.
const StatusBadge = ({ status }: { status: ToolCallPart['status'] }) => {
  const { theme } = useDesignSystemTheme();
  const icon =
    status === ToolCallStatus.Done ? (
      <CheckCircleIcon css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textValidationSuccess }} />
    ) : status === ToolCallStatus.Error ? (
      <XCircleIcon css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textValidationDanger }} />
    ) : (
      <Spinner size="small" />
    );
  return (
    <span
      data-testid={`tool-call-status-${status ?? ToolCallStatus.Running}`}
      css={{
        width: 16,
        height: 16,
        flexShrink: 0,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {icon}
    </span>
  );
};

const fencedBlock = (body: string, lang = ''): string => `\`\`\`${lang}\n${body}\n\`\`\``;

// Overall status for a run of tool calls. Running until every call resolves. Once settled,
// the run reflects how it *ended*: a failure that a later call recovered from (e.g. a retry)
// reads as done, not failed — only a trailing error surfaces as `error`. Individual failures
// stay visible on their own cards.
export const groupStatus = (parts: ToolCallPart[]): NonNullable<ToolCallPart['status']> => {
  if (parts.some((p) => (p.status ?? ToolCallStatus.Running) === ToolCallStatus.Running)) return ToolCallStatus.Running;
  return parts[parts.length - 1]?.status === ToolCallStatus.Error ? ToolCallStatus.Error : ToolCallStatus.Done;
};

// Deduped, first-appearance-ordered tool names with `×N` when a name repeats,
// e.g. [load_skill, Bash, Bash] → "load_skill, Bash ×2".
export const toolNameSummary = (parts: ToolCallPart[]): string => {
  const counts = new Map<string, number>();
  for (const part of parts) {
    counts.set(part.name, (counts.get(part.name) ?? 0) + 1);
  }
  return [...counts.entries()].map(([name, n]) => (n > 1 ? `${name} ×${n}` : name)).join(', ');
};

const GROUP_STATUS_LABEL: Record<NonNullable<ToolCallPart['status']>, ReactNode> = {
  [ToolCallStatus.Running]: (
    <FormattedMessage defaultMessage="Running" description="Status for an in-progress run of assistant tool calls" />
  ),
  [ToolCallStatus.Done]: (
    <FormattedMessage defaultMessage="Completed" description="Status for a finished run of assistant tool calls" />
  ),
  [ToolCallStatus.Error]: (
    <FormattedMessage defaultMessage="Failed" description="Status for a failed run of assistant tool calls" />
  ),
};

// Activate a role="button" disclosure row from the keyboard, matching native <button> semantics.
const onDisclosureKeyDown = (toggle: () => void) => (event: KeyboardEvent) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    toggle();
  }
};

/**
 * Renders a run of consecutive tool calls as one collapsible row: the header shows the
 * call count, a deduped tool-name summary, and an overall status; expanding reveals the
 * individual {@link ToolCallCard}s.
 */
export const ToolCallGroup = ({ parts }: { parts: ToolCallPart[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expanded, setExpanded] = useState(false);
  const status = groupStatus(parts);
  const summary = toolNameSummary(parts);
  const statusColor =
    status === ToolCallStatus.Done
      ? theme.colors.textValidationSuccess
      : status === ToolCallStatus.Error
        ? theme.colors.textValidationDanger
        : theme.colors.textSecondary;

  const toggle = () => setExpanded((prev) => !prev);

  return (
    <div css={{ margin: `${theme.spacing.md}px 0` }}>
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={intl.formatMessage({
          defaultMessage: 'Tool calls',
          description: 'Accessible label for the expandable summary row of a run of assistant tool calls',
        })}
        onClick={toggle}
        onKeyDown={onDisclosureKeyDown(toggle)}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          minWidth: 0,
        }}
      >
        {expanded ? (
          <ChevronDownIcon css={{ flexShrink: 0 }} aria-hidden />
        ) : (
          <ChevronRightIcon css={{ flexShrink: 0 }} aria-hidden />
        )}
        <WrenchSparkleIcon
          css={{ fontSize: theme.typography.fontSizeSm, flexShrink: 0, color: theme.colors.textSecondary }}
        />
        <Typography.Text size="sm" color="secondary" bold css={{ flexShrink: 0 }}>
          <FormattedMessage
            defaultMessage="{count, plural, one {# tool call} other {# tool calls}}"
            description="Count of tool calls the assistant made in a turn"
            values={{ count: parts.length }}
          />
        </Typography.Text>
        {summary && (
          <Typography.Text
            size="sm"
            color="secondary"
            css={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              minWidth: 0,
            }}
          >
            {summary}
          </Typography.Text>
        )}
        <Typography.Text
          size="sm"
          data-testid={`tool-group-status-${status}`}
          css={{ flexShrink: 0, marginLeft: 'auto', color: statusColor }}
        >
          {GROUP_STATUS_LABEL[status]}
        </Typography.Text>
      </div>

      {expanded && (
        <div
          css={{
            marginTop: theme.spacing.sm,
            marginLeft: theme.spacing.xs,
            paddingLeft: theme.spacing.md,
            borderLeft: `1px solid ${theme.colors.border}`,
          }}
        >
          {parts.map((part) => (
            <ToolCallCard key={part.toolUseId} part={part} />
          ))}
        </div>
      )}
    </div>
  );
};

export const ToolCallCard = ({ part }: { part: ToolCallPart }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expanded, setExpanded] = useState(false);
  const summary = toolInputSummary(part);
  // Only stringify the (potentially large) input when the card is expanded, so collapsed
  // cards stay cheap while scrolling long transcripts.
  const inputJson = useMemo(() => (expanded ? JSON.stringify(part.input ?? {}, null, 2) : ''), [expanded, part.input]);
  const toggle = () => setExpanded((prev) => !prev);

  return (
    <div css={{ margin: `${theme.spacing.sm}px 0` }}>
      <div
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={intl.formatMessage(
          {
            defaultMessage: 'Tool call: {name}',
            description: 'Accessible label for an expandable assistant tool-call card',
          },
          { name: part.name },
        )}
        onClick={toggle}
        onKeyDown={onDisclosureKeyDown(toggle)}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          minWidth: 0,
        }}
      >
        {expanded ? (
          <ChevronDownIcon css={{ flexShrink: 0 }} aria-hidden />
        ) : (
          <ChevronRightIcon css={{ flexShrink: 0 }} aria-hidden />
        )}
        <WrenchSparkleIcon
          css={{ fontSize: theme.typography.fontSizeSm, flexShrink: 0, color: theme.colors.textSecondary }}
        />
        <Typography.Text size="sm" color="secondary" bold css={{ flexShrink: 0 }}>
          {part.name}
        </Typography.Text>
        <StatusBadge status={part.status} />
        {summary && (
          <Typography.Text
            size="sm"
            color="secondary"
            css={{
              fontFamily: 'monospace',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              minWidth: 0,
            }}
          >
            {summary}
          </Typography.Text>
        )}
      </div>

      {expanded && (
        <div css={{ paddingLeft: theme.spacing.lg, marginTop: theme.spacing.xs }}>
          <Typography.Text size="sm" color="secondary" bold>
            <FormattedMessage defaultMessage="Input" description="Label for an assistant tool call's input" />
          </Typography.Text>
          <GenAIMarkdownRenderer>{fencedBlock(inputJson, 'json')}</GenAIMarkdownRenderer>
          {part.result != null && part.result !== '' && (
            <>
              <Typography.Text size="sm" color="secondary" bold>
                <FormattedMessage defaultMessage="Output" description="Label for an assistant tool call's output" />
              </Typography.Text>
              <GenAIMarkdownRenderer>{fencedBlock(truncate(part.result))}</GenAIMarkdownRenderer>
            </>
          )}
        </div>
      )}
    </div>
  );
};

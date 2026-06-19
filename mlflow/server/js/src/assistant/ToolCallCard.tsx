/**
 * Renders a single tool call the assistant made, inline in the transcript: a
 * collapsible card whose header shows the tool name, a status badge, and a one-line
 * input summary, and whose expanded body shows the full input and (truncated) output.
 */
import { useState, type ReactNode } from 'react';
import {
  Button,
  CheckCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  Spinner,
  Typography,
  WrenchSparkleIcon,
  XCircleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { AssistantPart } from './types';
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
    status === 'done' ? (
      <CheckCircleIcon css={{ fontSize: 14, color: theme.colors.textValidationSuccess }} />
    ) : status === 'error' ? (
      <XCircleIcon css={{ fontSize: 14, color: theme.colors.textValidationDanger }} />
    ) : (
      <Spinner size="small" />
    );
  return (
    <span
      data-testid={`tool-call-status-${status ?? 'running'}`}
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
  if (parts.some((p) => (p.status ?? 'running') === 'running')) return 'running';
  return parts[parts.length - 1]?.status === 'error' ? 'error' : 'done';
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
  running: (
    <FormattedMessage defaultMessage="Running" description="Status for an in-progress run of assistant tool calls" />
  ),
  done: <FormattedMessage defaultMessage="Completed" description="Status for a finished run of assistant tool calls" />,
  error: <FormattedMessage defaultMessage="Failed" description="Status for a failed run of assistant tool calls" />,
};

/**
 * Renders a run of consecutive tool calls as one collapsible row: the header shows the
 * call count, a deduped tool-name summary, and an overall status; expanding reveals the
 * individual {@link ToolCallCard}s.
 */
export const ToolCallGroup = ({ parts }: { parts: ToolCallPart[] }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const status = groupStatus(parts);
  const summary = toolNameSummary(parts);
  const statusColor =
    status === 'done'
      ? theme.colors.textValidationSuccess
      : status === 'error'
        ? theme.colors.textValidationDanger
        : theme.colors.textSecondary;

  return (
    <div css={{ margin: `${theme.spacing.md}px 0` }} aria-label="Tool calls">
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          minWidth: 0,
        }}
        onClick={() => setExpanded((prev) => !prev)}
      >
        <Button
          componentId="mlflow.assistant.chat_panel.tool_group.expand"
          size="small"
          type="tertiary"
          css={{ flexShrink: 0 }}
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
        />
        <WrenchSparkleIcon css={{ fontSize: 14, flexShrink: 0, color: theme.colors.textSecondary }} />
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
  const [expanded, setExpanded] = useState(false);
  const summary = toolInputSummary(part);
  const inputJson = JSON.stringify(part.input ?? {}, null, 2);

  return (
    <div css={{ margin: `${theme.spacing.sm}px 0` }} aria-label={`Tool call: ${part.name}`}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          minWidth: 0,
        }}
        onClick={() => setExpanded((prev) => !prev)}
      >
        <Button
          componentId="mlflow.assistant.chat_panel.tool_call.expand"
          size="small"
          type="tertiary"
          css={{ flexShrink: 0 }}
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
        />
        <WrenchSparkleIcon css={{ fontSize: 14, flexShrink: 0, color: theme.colors.textSecondary }} />
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

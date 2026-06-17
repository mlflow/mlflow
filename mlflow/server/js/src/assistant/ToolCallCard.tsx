/**
 * Renders a single tool call the assistant made, inline in the transcript: a
 * collapsible card whose header shows the tool name, a status badge, and a one-line
 * input summary, and whose expanded body shows the full input and (truncated) output.
 */
import { useState } from 'react';
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

// Input fields, in priority order, used to build the one-line summary in the header.
const SUMMARY_KEYS = ['command', 'trace_id', 'description', 'file_path'] as const;

// Tool output can be huge (e.g. a full trace); cap what we render in the card.
const MAX_OUTPUT_CHARS = 4000;

const truncate = (text: string): string =>
  text.length > MAX_OUTPUT_CHARS
    ? `${text.slice(0, MAX_OUTPUT_CHARS)}\n… (truncated, ${text.length - MAX_OUTPUT_CHARS} more chars)`
    : text;

// One-line, human-readable summary of a tool call's input for the header.
const toolInputSummary = (part: ToolCallPart): string => {
  const input = part.input ?? {};
  const traceId = input['trace_id'];
  const jqFilter = input['jq_filter'];
  if (typeof traceId === 'string') {
    return typeof jqFilter === 'string' && jqFilter ? `${traceId} · ${jqFilter}` : traceId;
  }
  for (const key of SUMMARY_KEYS) {
    const value = input[key];
    if (typeof value === 'string') return value;
  }
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

export const ToolCallCard = ({ part }: { part: ToolCallPart }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const summary = toolInputSummary(part);
  const inputJson = JSON.stringify(part.input ?? {}, null, 2);

  return (
    <div css={{ margin: `${theme.spacing.xs}px 0` }} aria-label={`Tool call: ${part.name}`}>
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

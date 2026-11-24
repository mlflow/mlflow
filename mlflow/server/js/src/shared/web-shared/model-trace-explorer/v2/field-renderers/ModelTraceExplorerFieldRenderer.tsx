import { every, isBoolean, isNumber, isString } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerAttachmentRenderer } from './ModelTraceExplorerAttachmentRenderer';
import { ModelTraceExplorerChatToolsRenderer } from './ModelTraceExplorerChatToolsRenderer';
import { ModelTraceExplorerRetrieverFieldRenderer } from './ModelTraceExplorerRetrieverFieldRenderer';
import { ModelTraceExplorerTextFieldRenderer } from './ModelTraceExplorerTextFieldRenderer';
import { containsAttachmentUri, parseAttachmentUri } from '../../attachment-utils';
import type { Assessment, SearchMatch } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { isModelTraceChatTool, isRetrieverDocument, normalizeConversation } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerConversation } from '../right-pane/ModelTraceExplorerConversation';

const MAX_ATTACHMENT_SEARCH_DEPTH = 10;
const MAX_LINE_DIFF_MATRIX_CELLS = 4_000_000;

interface UnifiedComparisonLine {
  kind: 'unchanged' | 'removed' | 'added';
  text: string;
}

const normalizeComparisonValue = (data: string | undefined): string | undefined => {
  if (data === undefined) {
    return undefined;
  }
  try {
    const sortKeys = (value: unknown): unknown => {
      if (Array.isArray(value)) {
        return value.map(sortKeys);
      }
      if (value && typeof value === 'object') {
        return Object.fromEntries(
          Object.entries(value)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([key, nestedValue]) => [key, sortKeys(nestedValue)]),
        );
      }
      return value;
    };
    return JSON.stringify(sortKeys(JSON.parse(data)), null, 2);
  } catch {
    return data;
  }
};

export const areModelTraceComparisonValuesEqual = (first: string | undefined, second: string | undefined): boolean =>
  normalizeComparisonValue(first) === normalizeComparisonValue(second);

const getCoarseUnifiedComparisonLines = (
  baselineLines: string[],
  candidateLines: string[],
): UnifiedComparisonLine[] => {
  let prefixLength = 0;
  while (
    prefixLength < baselineLines.length &&
    prefixLength < candidateLines.length &&
    baselineLines[prefixLength] === candidateLines[prefixLength]
  ) {
    prefixLength += 1;
  }
  let suffixLength = 0;
  while (
    suffixLength < baselineLines.length - prefixLength &&
    suffixLength < candidateLines.length - prefixLength &&
    baselineLines[baselineLines.length - suffixLength - 1] === candidateLines[candidateLines.length - suffixLength - 1]
  ) {
    suffixLength += 1;
  }
  return [
    ...baselineLines.slice(0, prefixLength).map((text) => ({ kind: 'unchanged' as const, text })),
    ...baselineLines
      .slice(prefixLength, baselineLines.length - suffixLength)
      .map((text) => ({ kind: 'removed' as const, text })),
    ...candidateLines
      .slice(prefixLength, candidateLines.length - suffixLength)
      .map((text) => ({ kind: 'added' as const, text })),
    ...(suffixLength > 0
      ? baselineLines.slice(baselineLines.length - suffixLength).map((text) => ({ kind: 'unchanged' as const, text }))
      : []),
  ];
};

const getUnifiedComparisonLines = (baseline: string | undefined, candidate: string | undefined) => {
  const baselineLines = baseline?.split('\n') ?? [];
  const candidateLines = candidate?.split('\n') ?? [];
  if (baselineLines.length * candidateLines.length > MAX_LINE_DIFF_MATRIX_CELLS) {
    return getCoarseUnifiedComparisonLines(baselineLines, candidateLines);
  }

  const rowLength = candidateLines.length + 1;
  const commonLineCounts = new Uint32Array((baselineLines.length + 1) * rowLength);
  for (let baselineIndex = baselineLines.length - 1; baselineIndex >= 0; baselineIndex -= 1) {
    for (let candidateIndex = candidateLines.length - 1; candidateIndex >= 0; candidateIndex -= 1) {
      const index = baselineIndex * rowLength + candidateIndex;
      commonLineCounts[index] =
        baselineLines[baselineIndex] === candidateLines[candidateIndex]
          ? commonLineCounts[(baselineIndex + 1) * rowLength + candidateIndex + 1] + 1
          : Math.max(
              commonLineCounts[(baselineIndex + 1) * rowLength + candidateIndex],
              commonLineCounts[baselineIndex * rowLength + candidateIndex + 1],
            );
    }
  }

  const lines: UnifiedComparisonLine[] = [];
  let baselineIndex = 0;
  let candidateIndex = 0;
  while (baselineIndex < baselineLines.length && candidateIndex < candidateLines.length) {
    if (baselineLines[baselineIndex] === candidateLines[candidateIndex]) {
      lines.push({ kind: 'unchanged', text: baselineLines[baselineIndex] });
      baselineIndex += 1;
      candidateIndex += 1;
    } else if (
      commonLineCounts[(baselineIndex + 1) * rowLength + candidateIndex] >=
      commonLineCounts[baselineIndex * rowLength + candidateIndex + 1]
    ) {
      lines.push({ kind: 'removed', text: baselineLines[baselineIndex] });
      baselineIndex += 1;
    } else {
      lines.push({ kind: 'added', text: candidateLines[candidateIndex] });
      candidateIndex += 1;
    }
  }
  while (baselineIndex < baselineLines.length) {
    lines.push({ kind: 'removed', text: baselineLines[baselineIndex] });
    baselineIndex += 1;
  }
  while (candidateIndex < candidateLines.length) {
    lines.push({ kind: 'added', text: candidateLines[candidateIndex] });
    candidateIndex += 1;
  }
  return lines;
};

export interface ModelTraceComparisonLineCounts {
  added: number;
  removed: number;
}

export const getModelTraceComparisonLineCounts = (
  baseline: string | undefined,
  candidate: string | undefined,
): ModelTraceComparisonLineCounts => {
  const lines = getUnifiedComparisonLines(normalizeComparisonValue(baseline), normalizeComparisonValue(candidate));
  return lines.reduce<ModelTraceComparisonLineCounts>(
    (counts, line) => ({
      added: counts.added + Number(line.kind === 'added'),
      removed: counts.removed + Number(line.kind === 'removed'),
    }),
    { added: 0, removed: 0 },
  );
};

/**
 * Recursively extract all attachment URIs from a parsed JSON value.
 */
export const findAttachmentUris = (
  value: unknown,
  depth = 0,
): { attachmentId: string; traceId: string; contentType: string; size?: number }[] => {
  if (depth > MAX_ATTACHMENT_SEARCH_DEPTH) {
    return [];
  }
  if (isString(value)) {
    if (!value.startsWith('mlflow-attachment://')) return [];
    const parsed = parseAttachmentUri(value);
    return parsed ? [parsed] : [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((v) => findAttachmentUris(v, depth + 1));
  }
  if (value && typeof value === 'object') {
    return Object.values(value).flatMap((v) => findAttachmentUris(v, depth + 1));
  }
  return [];
};

export const DEFAULT_MAX_VISIBLE_CHAT_MESSAGES = 3;

export const ModelTraceExplorerFieldRenderer = ({
  title,
  data,
  renderMode,
  chatMessageFormat,
  maxVisibleMessages = DEFAULT_MAX_VISIBLE_CHAT_MESSAGES,
  assessments,
  searchFilter,
  activeMatch,
  containsActiveMatch,
  comparisonData,
  comparisonRole,
  comparisonEnabled = false,
}: {
  title: string;
  data?: string;
  renderMode: 'default' | 'json' | 'text' | 'yaml';
  chatMessageFormat?: string;
  maxVisibleMessages?: number;
  assessments?: Assessment[];
  searchFilter?: string;
  activeMatch?: SearchMatch | null;
  containsActiveMatch?: boolean;
  comparisonData?: string;
  comparisonRole?: 'baseline' | 'candidate';
  comparisonEnabled?: boolean;
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();
  const [messagesExpanded, setMessagesExpanded] = useState(false);
  const parsedData = useMemo(() => {
    try {
      return data === undefined ? undefined : JSON.parse(data);
    } catch (e) {
      return data;
    }
  }, [data]);

  useEffect(() => {
    setMessagesExpanded(false);
  }, [data]);

  if (comparisonEnabled && comparisonRole) {
    const baseline = normalizeComparisonValue(comparisonRole === 'baseline' ? data : comparisonData);
    const candidate = normalizeComparisonValue(comparisonRole === 'candidate' ? data : comparisonData);
    if (!areModelTraceComparisonValuesEqual(baseline, candidate)) {
      const lines = getUnifiedComparisonLines(baseline, candidate);
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {title && (
            <Typography.Text bold color="secondary" size="sm">
              {title}
            </Typography.Text>
          )}
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusSm,
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {baseline === undefined && (
              <div
                css={{
                  padding: theme.spacing.xs,
                  backgroundColor: theme.colors.backgroundSecondary,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage
                  defaultMessage="Baseline: missing"
                  description="Label for an absent baseline value in a trace comparison diff"
                />
              </div>
            )}
            {candidate === undefined && (
              <div
                css={{
                  padding: theme.spacing.xs,
                  backgroundColor: theme.colors.backgroundSecondary,
                  color: theme.colors.textSecondary,
                }}
              >
                <FormattedMessage
                  defaultMessage="Candidate: missing"
                  description="Label for an absent candidate value in a trace comparison diff"
                />
              </div>
            )}
            {lines.map((line, index) => (
              <div
                key={`${line.kind}-${index}`}
                css={{
                  paddingInline: theme.spacing.xs,
                  backgroundColor:
                    line.kind === 'removed'
                      ? theme.colors.backgroundValidationDanger
                      : line.kind === 'added'
                        ? theme.colors.backgroundValidationSuccess
                        : 'transparent',
                  color:
                    line.kind === 'removed'
                      ? theme.colors.textValidationDanger
                      : line.kind === 'added'
                        ? theme.colors.textValidationSuccess
                        : theme.colors.textPrimary,
                }}
              >
                {line.kind === 'removed' ? '− ' : line.kind === 'added' ? '+ ' : '  '}
                {line.text || ' '}
              </div>
            ))}
          </div>
        </div>
      );
    }
  }

  const dataIsScalar = isString(parsedData) || isNumber(parsedData) || isBoolean(parsedData);
  // wrap the value in an object with the title as key. this helps normalizeConversation
  // recognize the format, as this util function was designed to receive the whole input
  // object rather than value by value. it does not work for complex cases where we need
  // to check multiple keys in the object (e.g. anthropic), but works for cases where we're
  // basically just looking for the field that contains chat messages.
  const chatMessages = normalizeConversation(title ? { [title]: parsedData } : parsedData, chatMessageFormat);
  const isChatTools = Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isModelTraceChatTool);
  const isRetrieverDocuments =
    Array.isArray(parsedData) && parsedData.length > 0 && every(parsedData, isRetrieverDocument);

  if (chatMessages && chatMessages.length > 0 && renderMode === 'default') {
    const shouldTruncateMessages = chatMessages.length > maxVisibleMessages;
    const visibleMessages =
      messagesExpanded || !shouldTruncateMessages ? chatMessages : chatMessages.slice(-maxVisibleMessages);
    const hiddenMessageCount = shouldTruncateMessages ? chatMessages.length - visibleMessages.length : 0;

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          padding: 0,
          borderRadius: 0,
        }}
      >
        {title && (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingInline: 0,
            }}
          >
            <Typography.Text bold color="secondary" size="sm">
              {title}
            </Typography.Text>
          </div>
        )}
        {shouldTruncateMessages && (
          <Typography.Link
            css={{ alignSelf: 'flex-start', marginLeft: theme.spacing.xs }}
            componentId="shared.model-trace-explorer.conversation-toggle"
            onClick={() => setMessagesExpanded(!messagesExpanded)}
          >
            {messagesExpanded ? (
              <FormattedMessage
                defaultMessage="Show less"
                description="Button label to collapse conversation messages in model trace explorer"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Show {hiddenMessageCount} more"
                description="Button label to expand and show hidden conversation messages in model trace explorer"
                values={{ hiddenMessageCount }}
              />
            )}
          </Typography.Link>
        )}
        <ModelTraceExplorerConversation messages={visibleMessages} />
      </div>
    );
  }

  if (renderMode === 'json') {
    return (
      <ModelTraceExplorerCodeSnippet
        title={title}
        data={data ?? ''}
        initialRenderMode={CodeSnippetRenderMode.JSON}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        containsActiveMatch={containsActiveMatch}
      />
    );
  }

  if (renderMode === 'text') {
    return (
      <ModelTraceExplorerCodeSnippet
        title={title}
        data={data ?? ''}
        initialRenderMode={CodeSnippetRenderMode.TEXT}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        containsActiveMatch={containsActiveMatch}
      />
    );
  }

  if (renderMode === 'yaml') {
    return (
      <ModelTraceExplorerCodeSnippet
        title={title}
        data={data ?? ''}
        initialRenderMode={CodeSnippetRenderMode.YAML}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        containsActiveMatch={containsActiveMatch}
      />
    );
  }

  if (dataIsScalar) {
    if (isString(parsedData)) {
      const attachment = parseAttachmentUri(parsedData);
      if (attachment) {
        return (
          <ModelTraceExplorerAttachmentRenderer
            title={title}
            attachmentId={attachment.attachmentId}
            traceId={attachment.traceId}
            contentType={attachment.contentType}
            size={attachment.size}
          />
        );
      }
    }
    return <ModelTraceExplorerTextFieldRenderer title={title} value={String(parsedData)} />;
  }

  if (isChatTools) {
    return <ModelTraceExplorerChatToolsRenderer title={title} tools={parsedData} />;
  }

  if (isRetrieverDocuments) {
    return <ModelTraceExplorerRetrieverFieldRenderer title={title} documents={parsedData} assessments={assessments} />;
  }

  // Check for attachment URIs embedded in complex JSON structures
  if (containsAttachmentUri(data ?? '')) {
    const attachments = findAttachmentUris(parsedData);
    if (attachments.length > 0) {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {attachments.map((att) => (
            <ModelTraceExplorerAttachmentRenderer
              key={att.attachmentId}
              title=""
              attachmentId={att.attachmentId}
              traceId={att.traceId}
              contentType={att.contentType}
              size={att.size}
            />
          ))}
          <ModelTraceExplorerCodeSnippet
            title={title}
            data={data ?? ''}
            initialRenderMode={CodeSnippetRenderMode.YAML}
          />
        </div>
      );
    }
  }

  return (
    <ModelTraceExplorerCodeSnippet title={title} data={data ?? ''} initialRenderMode={CodeSnippetRenderMode.YAML} />
  );
};

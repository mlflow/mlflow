import { useCallback, useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useModelTraceExplorerViewState } from './ModelTraceExplorerViewStateContext';
import { searchTreeBySpanId } from './ModelTraceExplorer.utils';

/**
 * Regex to match markdown-style links: [text](path)
 * Captures: [1] = link text, [2] = path
 */
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g;

/**
 * Matches a fully-qualified deeplink path:
 *   /experiments/{experimentId}/traces/{traceId}/spans/{spanId}
 */
const FULL_PATH_RE = /^\/experiments\/([^/]+)\/traces\/([^/]+)\/spans\/([^/]+)$/;

/**
 * Matches a relative deeplink path:
 *   spans/{spanId}
 */
const RELATIVE_PATH_RE = /^spans\/([^/]+)$/;

interface ParsedDeeplink {
  experimentId: string;
  traceId: string;
  spanId: string;
}

function parseDeeplinkPath(path: string): ParsedDeeplink | null {
  const fullMatch = FULL_PATH_RE.exec(path);
  if (fullMatch) {
    return { experimentId: fullMatch[1], traceId: fullMatch[2], spanId: fullMatch[3] };
  }
  const relativeMatch = RELATIVE_PATH_RE.exec(path);
  if (relativeMatch) {
    return { experimentId: '', traceId: '', spanId: relativeMatch[1] };
  }
  return null;
}

interface TextSegment {
  type: 'text';
  content: string;
}

interface DeeplinkSegment {
  type: 'deeplink';
  text: string;
  spanId: string;
  experimentId: string;
  traceId: string;
}

type Segment = TextSegment | DeeplinkSegment;

function parseDescription(text: string): Segment[] {
  const segments: Segment[] = [];
  let lastIndex = 0;

  // Reset regex state
  MARKDOWN_LINK_RE.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = MARKDOWN_LINK_RE.exec(text)) !== null) {
    // Add preceding text
    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: text.slice(lastIndex, match.index) });
    }

    const linkText = match[1];
    const path = match[2];
    const parsed = parseDeeplinkPath(path);

    if (parsed) {
      segments.push({
        type: 'deeplink',
        text: linkText,
        spanId: parsed.spanId,
        experimentId: parsed.experimentId,
        traceId: parsed.traceId,
      });
    } else {
      // Not a valid deeplink — render as plain text
      segments.push({ type: 'text', content: match[0] });
    }

    lastIndex = match.index + match[0].length;
  }

  // Add trailing text
  if (lastIndex < text.length) {
    segments.push({ type: 'text', content: text.slice(lastIndex) });
  }

  return segments;
}

/**
 * Renders description text with inline deeplinks to spans.
 *
 * Deeplink syntax (markdown-style):
 *   [link text](/experiments/{experimentId}/traces/{traceId}/spans/{spanId})
 *   [link text](spans/{spanId})   — relative, resolved against current trace
 *
 * Clicking a deeplink selects the target span in the timeline tree and scrolls
 * it into view. Cross-trace deeplinks are rendered but not yet navigable.
 */
export const DeeplinkText = ({
  text,
  size = 'md',
  color,
}: {
  text: string;
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'secondary';
}) => {
  const { theme } = useDesignSystemTheme();
  const { rootNode, setSelectedNode, setActiveView, topLevelNodes } = useModelTraceExplorerViewState();
  const currentTraceId = rootNode?.traceId ?? '';

  const segments = useMemo(() => parseDescription(text), [text]);

  const handleDeeplinkClick = useCallback(
    (segment: DeeplinkSegment) => {
      // For relative links or same-trace links, select the span in-context
      const isCurrentTrace = !segment.traceId || segment.traceId === currentTraceId;
      if (isCurrentTrace) {
        // Search across all top-level nodes (handles multi-root traces)
        for (const node of topLevelNodes) {
          const found = searchTreeBySpanId(node, segment.spanId);
          if (found) {
            setSelectedNode(found);
            // Switch to detail view so the timeline tree is visible
            setActiveView('detail');
            // Scroll the timeline tree node into view after the view switch renders
            setTimeout(() => {
              const el = document.querySelector(`[data-testid="timeline-tree-node-${segment.spanId}"]`);
              el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
            return;
          }
        }
      }
      // Cross-trace: navigate via URL
      if (!isCurrentTrace && segment.experimentId) {
        const url = `/experiments/${segment.experimentId}/traces?selectedTraceId=${segment.traceId}&focusSpan=${segment.spanId}`;
        window.location.href = url;
      }
    },
    [currentTraceId, topLevelNodes, setSelectedNode, setActiveView],
  );

  if (segments.length === 1 && segments[0].type === 'text') {
    return (
      <Typography.Text size={size} color={color}>
        {segments[0].content}
      </Typography.Text>
    );
  }

  return (
    <Typography.Text size={size} color={color}>
      {segments.map((segment, i) => {
        if (segment.type === 'text') {
          return <span key={i}>{segment.content}</span>;
        }
        return (
          <span
            key={i}
            role="link"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              handleDeeplinkClick(segment);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.stopPropagation();
                handleDeeplinkClick(segment);
              }
            }}
            css={{
              color: theme.colors.actionPrimaryBackgroundDefault,
              cursor: 'pointer',
              textDecoration: 'underline',
              textDecorationColor: 'transparent',
              transition: 'text-decoration-color 150ms ease',
              ':hover': {
                textDecorationColor: theme.colors.actionPrimaryBackgroundDefault,
              },
            }}
          >
            {segment.text}
          </span>
        );
      })}
    </Typography.Text>
  );
};

// Re-export for testing
export { parseDescription, parseDeeplinkPath };
export type { Segment, DeeplinkSegment, TextSegment, ParsedDeeplink };

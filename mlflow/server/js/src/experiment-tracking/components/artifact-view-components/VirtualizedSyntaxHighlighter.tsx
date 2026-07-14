import React, { useCallback, useMemo, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { useVirtualizer, type VirtualItem } from '@tanstack/react-virtual';
import { createElement } from 'react-syntax-highlighter';
import type { SyntaxHighlighterProps } from 'react-syntax-highlighter';

type rendererNode = {
  type: 'element' | 'text';
  value?: string | number;
  tagName?: any;
  properties?: any;
  children?: rendererNode[];
};

const ESTIMATED_LINE_HEIGHT = 20;
const OVERSCAN_LINES = 10;

/**
 * Counts lines without allocating one array entry per line (unlike
 * `text.split('\n').length`), which matters for very large artifacts.
 * Matches split semantics: an empty string counts as one line.
 */
export function countLines(text: string): number {
  let lines = 1;
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) === 10 /* \n */) {
      lines++;
    }
  }
  return lines;
}

export interface VirtualizedSyntaxHighlighterProps extends Omit<
  SyntaxHighlighterProps,
  'renderer' | 'PreTag' | 'CodeTag'
> {
  children: string;
}

/**
 * Strips the trailing newline from the deepest last text node of a highlighted row.
 * react-syntax-highlighter ends every line node with a "\n" to drive line breaks
 * when rows are rendered as siblings. When we render each row in its own
 * absolutely-positioned container, that trailing newline would create an extra
 * empty line, so we remove it.
 */
function stripTrailingNewline(node: rendererNode): rendererNode {
  if (node.type === 'text') {
    return { ...node, value: String(node.value ?? '').replace(/\n$/, '') };
  }

  if (!node.children || node.children.length === 0) {
    return node;
  }

  const lastIndex = node.children.length - 1;
  const lastChild = node.children[lastIndex];

  return {
    ...node,
    children: [...node.children.slice(0, lastIndex), stripTrailingNewline(lastChild)],
  };
}

/**
 * A virtualized wrapper around react-syntax-highlighter for large multi-line files.
 *
 * react-syntax-highlighter crashes on very large files because its internal
 * processLines() spreads every line as an argument to [].concat(), exceeding
 * the JS engine argument limit. Supplying a custom renderer forces wrapLines
 * to true, which bypasses that code path. Combined with @tanstack/react-virtual,
 * we render only the lines near the viewport so the UI stays responsive.
 */
export const VirtualizedSyntaxHighlighter = ({
  children,
  customStyle,
  ...syntaxHighlighterProps
}: VirtualizedSyntaxHighlighterProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const lineCount = useMemo(() => countLines(children), [children]);

  const virtualizer = useVirtualizer({
    count: lineCount,
    estimateSize: () => ESTIMATED_LINE_HEIGHT,
    getScrollElement: () => containerRef.current,
    overscan: OVERSCAN_LINES,
    measureElement:
      typeof window !== 'undefined' && navigator.userAgent.indexOf('Firefox') === -1
        ? (element) => element?.getBoundingClientRect().height
        : undefined,
  });

  const renderer = useCallback(
    ({ rows, stylesheet, useInlineStyles }: { rows: rendererNode[]; stylesheet: any; useInlineStyles: boolean }) => {
      const virtualItems = virtualizer.getVirtualItems();
      const totalSize = virtualizer.getTotalSize();

      return (
        <div
          data-testid="virtualized-syntax-highlighter-spacer"
          style={{
            height: totalSize,
            position: 'relative',
          }}
        >
          {virtualItems.map((virtualRow: VirtualItem<HTMLDivElement>) => (
            <div
              key={virtualRow.key}
              data-index={virtualRow.index}
              ref={(node) => virtualizer.measureElement(node)}
              style={{
                position: 'absolute',
                top: 0,
                transform: `translateY(${virtualRow.start}px)`,
                width: '100%',
              }}
            >
              {createElement({
                node: stripTrailingNewline(rows[virtualRow.index]),
                stylesheet,
                useInlineStyles,
                key: virtualRow.key,
              })}
            </div>
          ))}
        </div>
      );
    },
    [virtualizer],
  );

  const virtualizedCustomStyle = useMemo(
    () => ({
      ...customStyle,
      height: 'auto',
      overflow: 'visible',
    }),
    [customStyle],
  );

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        overflow: 'auto',
      }}
    >
      <SyntaxHighlighter
        PreTag="div"
        CodeTag="div"
        customStyle={virtualizedCustomStyle}
        renderer={renderer}
        {...syntaxHighlighterProps}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
};

export default VirtualizedSyntaxHighlighter;

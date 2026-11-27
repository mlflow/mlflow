import React, { useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import createElement from 'react-syntax-highlighter/dist/cjs/create-element';

type RendererProps = {
  rows: any[];
  stylesheet: Record<string, React.CSSProperties>;
  useInlineStyles: boolean;
};

// Initial height estimate only. Actual row heights are measured via the virtualizer's
// measureElement, so rows whose real height differs (theme/font-size changes, wrapped
// long lines, etc.) are sized correctly instead of clipped or overlapped.
const ESTIMATED_ROW_HEIGHT = 20;
const OVERSCAN = 50;

export const ARTIFACT_TEXT_VIRTUALIZED_SPACER_TESTID = 'artifact-text-virtualized-spacer';
export const ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID = 'artifact-text-virtualized-row';

function VirtualizedRows({ rows, stylesheet, useInlineStyles }: RendererProps) {
  // Track the scroll element in state (not a plain ref) so attaching it triggers a re-render.
  // This renderer's subtree mounts inside react-syntax-highlighter's <pre> before the scroll
  // container's ref attaches; useVirtualizer reads getScrollElement() during a layout effect and
  // only re-reads it on a later render, so a ref read directly would resolve to null on the first
  // pass and leave the virtualizer permanently unmeasured (visible only in a real browser, not in
  // jsdom tests that mock the virtualizer).
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => ESTIMATED_ROW_HEIGHT,
    overscan: OVERSCAN,
  });

  return (
    <div
      ref={setScrollElement}
      style={{
        width: '100%',
        height: '100%',
        overflow: 'auto',
      }}
    >
      <div
        data-testid={ARTIFACT_TEXT_VIRTUALIZED_SPACER_TESTID}
        style={{
          height: virtualizer.getTotalSize(),
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((item) => (
          <div
            key={item.key}
            data-index={item.index}
            data-testid={ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID}
            ref={virtualizer.measureElement}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${item.start}px)`,
            }}
          >
            {createElement({
              node: rows[item.index],
              stylesheet,
              useInlineStyles,
              key: `code-segment-${item.index}`,
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * A stable `react-syntax-highlighter` renderer that virtualizes rows so only the visible
 * lines are mounted. Supplying a renderer also makes react-syntax-highlighter set
 * `wrapLines=true`, which bypasses the crashing `[].concat(...lines)` path in `processLines()`.
 *
 * Exported as a single stable function (not a factory) so the `renderer` prop identity stays
 * constant across re-renders of the consuming component.
 */
export const virtualizedRenderer = ({ rows, stylesheet, useInlineStyles }: RendererProps) => (
  <VirtualizedRows rows={rows} stylesheet={stylesheet} useInlineStyles={useInlineStyles} />
);

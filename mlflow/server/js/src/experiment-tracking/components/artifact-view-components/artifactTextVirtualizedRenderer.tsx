import React, { useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import createElement from 'react-syntax-highlighter/dist/cjs/create-element';

type RendererProps = {
  rows: any[];
  stylesheet: Record<string, React.CSSProperties>;
  useInlineStyles: boolean;
};

const ROW_HEIGHT = 20;
const OVERSCAN = 50;

function VirtualizedRows({ rows, stylesheet, useInlineStyles }: RendererProps) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: OVERSCAN,
  });

  return (
    <div
      ref={parentRef}
      style={{
        width: '100%',
        height: '100%',
        overflow: 'auto',
      }}
    >
      <div
        style={{
          height: virtualizer.getTotalSize(),
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((item) => (
          <div
            key={item.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: item.size,
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

export function virtualizedRenderer() {
  return ({ rows, stylesheet, useInlineStyles }: RendererProps) => (
    <VirtualizedRows rows={rows} stylesheet={stylesheet} useInlineStyles={useInlineStyles} />
  );
}

import type { CSSProperties } from 'react';
export const SPAN_INDENT_WIDTH = 16;
export const TimelineTreeZIndex = { HIGH: 5, NORMAL: 3, LOW: 1 };

export const getLineCenterX = (depth: number) => depth * SPAN_INDENT_WIDTH + SPAN_INDENT_WIDTH / 2;

export const hierarchyLineStyles = (
  lines: { shouldRender: boolean; isActive: boolean }[],
  activeColor: string,
  inactiveColor: string,
): CSSProperties | undefined => {
  const visible = lines.flatMap((line, depth) => (line.shouldRender ? [{ ...line, depth }] : []));
  if (!visible.length) return undefined;
  const last = lines[lines.length - 1];
  return {
    position: 'absolute',
    inset: 0,
    backgroundImage: visible
      .map(({ isActive }) => {
        const color = isActive ? activeColor : inactiveColor;
        return `linear-gradient(${color}, ${color})`;
      })
      .join(', '),
    backgroundPosition: visible.map(({ depth }) => `${getLineCenterX(depth)}px 0`).join(', '),
    backgroundSize: '1px 100%',
    backgroundRepeat: 'no-repeat',
    // Only the last straight line crosses a connector.
    zIndex: last.shouldRender && last.isActive ? TimelineTreeZIndex.NORMAL : TimelineTreeZIndex.LOW,
  };
};

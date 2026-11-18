import { type RefObject, useEffect, useLayoutEffect, useRef } from 'react';

interface HorizontalScrollAnchor {
  header: Element;
  /** The header's left edge relative to the scroll viewport's left edge. */
  offset: number;
}

const findTableScroller = (container: HTMLElement) => container.querySelector<HTMLElement>('[role="table"]');

// The leftmost column header still in view. A table scrolled to its start has no anchor, so columns
// added before the visible ones push them right, as they do on first load.
const findHorizontalScrollAnchor = (scroller: HTMLElement): HorizontalScrollAnchor | undefined => {
  if (scroller.scrollLeft <= 0) {
    return undefined;
  }
  const viewportLeft = scroller.getBoundingClientRect().left;
  for (const header of scroller.querySelectorAll('[role="columnheader"]')) {
    const { left, right } = header.getBoundingClientRect();
    if (right > viewportLeft) {
      return { header, offset: left - viewportLeft };
    }
  }
  return undefined;
};

/**
 * Keeps a horizontally scrolled table's leftmost visible column in place when columns are added or
 * removed before it. Browsers only anchor scrolling on the block axis, and assessment columns load after
 * the rows and are inserted before the trailing columns, so without this the columns the user scrolled
 * to, and any header menu open on them, jump out of view.
 *
 * `columnsKey` must change whenever the rendered columns change.
 */
export const useTableHorizontalScrollAnchor = (containerRef: RefObject<HTMLElement>, columnsKey: string) => {
  const anchorRef = useRef<HorizontalScrollAnchor>();
  const lastScrollLeftRef = useRef<number>();

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }
    const scroller = findTableScroller(container);
    if (!scroller) {
      return undefined;
    }
    const recordCurrentAnchor = () => {
      lastScrollLeftRef.current = scroller.scrollLeft;
      anchorRef.current = findHorizontalScrollAnchor(scroller);
    };
    // Scroll events don't bubble, so listen in the capture phase for the table's own scroller.
    const recordAnchorAfterHorizontalScroll = (event: Event) => {
      if (event.target === scroller && scroller.scrollLeft !== lastScrollLeftRef.current) {
        recordCurrentAnchor();
      }
    };
    container.addEventListener('scroll', recordAnchorAfterHorizontalScroll, { capture: true, passive: true });

    // A resized viewport or column can move the anchor without scrolling. Refresh the stored offset
    // so the next column-set change compensates only for that change, not for an earlier resize.
    const resizeObserver =
      typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(() => recordCurrentAnchor());
    resizeObserver?.observe(scroller);
    for (const header of scroller.querySelectorAll('[role="columnheader"]')) {
      resizeObserver?.observe(header);
    }

    return () => {
      container.removeEventListener('scroll', recordAnchorAfterHorizontalScroll, { capture: true });
      resizeObserver?.disconnect();
    };
  }, [containerRef, columnsKey]);

  useLayoutEffect(() => {
    const container = containerRef.current;
    const scroller = container ? findTableScroller(container) : null;
    if (!scroller) {
      return;
    }
    const anchor = anchorRef.current;
    if (anchor && scroller.contains(anchor.header)) {
      const shift = anchor.header.getBoundingClientRect().left - scroller.getBoundingClientRect().left - anchor.offset;
      if (shift !== 0) {
        scroller.scrollLeft += shift;
      }
    }
    anchorRef.current = findHorizontalScrollAnchor(scroller);
    lastScrollLeftRef.current = scroller.scrollLeft;
  }, [containerRef, columnsKey]);
};

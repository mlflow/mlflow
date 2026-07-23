import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import type { RefObject } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { create } from '@databricks/web-shared/zustand';

interface FloatingObstructionStore {
  /** id -> width (px) reserved on the right edge by an open surface (e.g. a drawer). */
  obstructions: Record<string, number>;
  /** id -> height (px) reserved on the bottom edge by a pinned surface (e.g. an action bar). */
  bottomObstructions: Record<string, number>;
  setObstruction: (id: string, reservedRightPx: number) => void;
  removeObstruction: (id: string) => void;
  setBottomObstruction: (id: string, reservedBottomPx: number) => void;
  removeBottomObstruction: (id: string) => void;
}

const useFloatingObstructionStore = create<FloatingObstructionStore>((set) => ({
  obstructions: {},
  bottomObstructions: {},
  setObstruction: (id, reservedRightPx) =>
    set((state) => ({ obstructions: { ...state.obstructions, [id]: reservedRightPx } })),
  removeObstruction: (id) =>
    set((state) => {
      if (!(id in state.obstructions)) {
        return state;
      }
      const next = { ...state.obstructions };
      delete next[id];
      return { obstructions: next };
    }),
  setBottomObstruction: (id, reservedBottomPx) =>
    set((state) => {
      // No-op when unchanged so the per-render measure (useLayoutEffect) doesn't churn the store.
      if (state.bottomObstructions[id] === reservedBottomPx) {
        return state;
      }
      return { bottomObstructions: { ...state.bottomObstructions, [id]: reservedBottomPx } };
    }),
  removeBottomObstruction: (id) =>
    set((state) => {
      if (!(id in state.bottomObstructions)) {
        return state;
      }
      const next = { ...state.bottomObstructions };
      delete next[id];
      return { bottomObstructions: next };
    }),
}));

/**
 * Register that some surface (e.g. a right-side drawer) is currently occupying the right
 * edge of the viewport, so the floating Assistant button can move clear of it instead of
 * overlapping content. Pass the width (px) reserved on the right while the surface is open,
 * or 0 when it isn't. Multiple surfaces compose; the button uses the widest reservation.
 */
export const useRegisterFloatingObstruction = (reservedRightPx: number) => {
  const id = useMemo(() => uuidv4(), []);
  const setObstruction = useFloatingObstructionStore((state) => state.setObstruction);
  const removeObstruction = useFloatingObstructionStore((state) => state.removeObstruction);
  useEffect(() => {
    if (!reservedRightPx || reservedRightPx <= 0) {
      removeObstruction(id);
      return undefined;
    }
    setObstruction(id, reservedRightPx);
    return () => removeObstruction(id);
  }, [id, reservedRightPx, setObstruction, removeObstruction]);
};

/** The widest reservation (px) currently held on the right edge, or 0 if none. */
export const useFloatingObstructionWidth = (): number =>
  useFloatingObstructionStore((state) => {
    const widths = Object.values(state.obstructions);
    return widths.length > 0 ? Math.max(...widths) : 0;
  });

// Rough height of the corner zone the resting FAB occupies (its bottom inset + a control-sized
// bubble + slack). A bar counts as obstructing only when its bottom edge reaches down into this
// zone — i.e. it is genuinely pinned near the viewport bottom. A `position: sticky; bottom: 0`
// footer floats mid-page while its scroll container has slack (bottom far above this zone) and
// drops into it only once the content overflows and the footer sticks; a footer anchored to a
// full-height container is always in the zone. This cleanly separates "pinned" from "floating"
// without depending on the page's exact bottom inset.
const FAB_CORNER_ZONE_PX = 64;

/**
 * Register that some surface (e.g. a bottom-pinned action bar) is currently occupying the
 * bottom edge of the viewport, so the floating Assistant button can rise clear of it instead
 * of overlapping its buttons. Pass a ref to the surface's outer element; while its bottom edge
 * sits within the FAB's corner zone it reserves the distance from its TOP up to the viewport
 * bottom (`innerHeight - rect.top`) — not its raw height, so the button clears the bar's top
 * even when the bar's scroll container is inset above `bottom: 0`. Re-measures on resize
 * (content-driven bars can wrap) and on scroll (a sticky footer pins/unpins as its container
 * scrolls). Deregisters automatically on unmount or when it floats up off the bottom.
 *
 * Ref-based (not a raw px) on purpose: the bottom bars have dynamic heights (wrapping status
 * text, conditional padding) and variable insets, so a hardcoded number would drift out of
 * date. The store still holds a number internally, symmetric with the right-edge axis.
 */
export const useRegisterFloatingBottomObstruction = (ref: RefObject<HTMLElement>) => {
  const id = useMemo(() => uuidv4(), []);
  const setBottomObstruction = useFloatingObstructionStore((state) => state.setBottomObstruction);
  const removeBottomObstruction = useFloatingObstructionStore((state) => state.removeBottomObstruction);

  const measure = useCallback(() => {
    const element = ref.current;
    if (!element) {
      removeBottomObstruction(id);
      return;
    }
    const rect = element.getBoundingClientRect();
    const pinnedToBottom = rect.bottom >= window.innerHeight - FAB_CORNER_ZONE_PX;
    // Reserve up to the bar's top so the FAB clears it regardless of any inset below the bar.
    const reservedBottomPx = window.innerHeight - rect.top;
    if (pinnedToBottom && rect.height > 0 && reservedBottomPx > 0) {
      setBottomObstruction(id, reservedBottomPx);
    } else {
      removeBottomObstruction(id);
    }
  }, [id, ref, setBottomObstruction, removeBottomObstruction]);

  // Re-measure on every render (pre-paint, so the button never flashes overlapping). A sticky
  // footer pins/unpins when *sibling* content grows or shrinks — e.g. adding a Playground
  // message pushes the footer down until it sticks. That changes the bar's position but not its
  // size and fires no scroll/resize event, so ResizeObserver and the listeners below can't see
  // it; the host component re-renders on that content change, so a layout effect is what
  // reliably catches it. measure() no-ops in the store when the reserved value is unchanged.
  useLayoutEffect(measure);

  // Backstops for changes that happen without a re-render of this component: the bar resizing
  // its own content, and viewport scroll/resize.
  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return undefined;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    // Capture-phase scroll so we also see scrolling inside nested scroll containers (a sticky
    // footer lives in one), not just window scroll.
    window.addEventListener('scroll', measure, true);
    window.addEventListener('resize', measure);
    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', measure, true);
      window.removeEventListener('resize', measure);
      removeBottomObstruction(id);
    };
  }, [id, ref, measure, removeBottomObstruction]);
};

/** The tallest reservation (px) currently held on the bottom edge, or 0 if none. */
export const useFloatingObstructionHeight = (): number =>
  useFloatingObstructionStore((state) => {
    const heights = Object.values(state.bottomObstructions);
    return heights.length > 0 ? Math.max(...heights) : 0;
  });

/**
 * AssistantAwareActionBar: a transparent wrapper for a bottom-pinned action bar (e.g. a
 * Cancel/Create or Cancel/Save footer) that registers its height with the floating-obstruction
 * store, so the Assistant floating button rises above it instead of overlapping its buttons.
 *
 * It is deliberately layout-neutral: it renders a single element, imposes no positioning of its
 * own, and forwards the caller's `css`/props unchanged. Callers keep whatever pins the bar
 * (`position: sticky; bottom: 0`, a flex `flexShrink: 0` footer, etc.) — this only measures the
 * rendered element (via ResizeObserver, so wrapping status text or conditional padding stays in
 * sync) and reports its height while mounted. When the bar unmounts (e.g. a footer that only
 * renders while a form has changes), the reservation is released automatically.
 *
 * Mirrors the AssistantAwareDrawer convention on the vertical axis.
 */
import { forwardRef, useImperativeHandle, useRef } from 'react';
import type { ComponentPropsWithoutRef } from 'react';
import { useRegisterFloatingBottomObstruction } from '../../assistant/useFloatingObstruction';

export const AssistantAwareActionBar = forwardRef<HTMLDivElement, ComponentPropsWithoutRef<'div'>>(
  function AssistantAwareActionBar({ children, ...rest }, forwardedRef) {
    const ref = useRef<HTMLDivElement>(null);
    // Keep our own ref for measuring while still honoring a caller-provided ref.
    useImperativeHandle(forwardedRef, () => ref.current as HTMLDivElement);
    useRegisterFloatingBottomObstruction(ref);
    return (
      <div ref={ref} {...rest}>
        {children}
      </div>
    );
  },
);

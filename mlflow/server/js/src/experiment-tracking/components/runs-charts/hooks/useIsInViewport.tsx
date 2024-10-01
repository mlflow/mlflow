import { useCallback, useEffect, useRef, useState } from 'react';
import { shouldDeferLineChartRendering } from '../../../../common/utils/FeatureUtils';
import { useDebounce } from 'use-debounce';

const VIEWPORT_DEBOUNCE_MS = 150;

/**
 * Checks if the element is currently visible within the viewport using IntersectionObserver.
 * If "enabled" is set to false, the returned value will always be true.
 */
export const useIsInViewport = ({
  enabled = true,
  deferTimeoutMs = VIEWPORT_DEBOUNCE_MS,
}: { enabled?: boolean; deferTimeoutMs?: number } = {}) => {
  const activeObserver = useRef<IntersectionObserver | null>(null);
  const [isInViewport, setIsInViewport] = useState(!enabled);
  // Let's use IntersectionObserver to determine if the element is displayed within the viewport
  const startObserver = useCallback(
    (element: Element | null) => {
      // If "enabled" is set to false or IntersectionObserver is not available, assume that the element is visible
      if (!enabled || !window.IntersectionObserver) {
        setIsInViewport(true);
        return () => {};
      }

      if (!element) {
        return () => {};
      }

      // Set the state flag only if element is in viewport
      const intersectionObserver = new IntersectionObserver(([entry]) => {
        setIsInViewport(entry.isIntersecting);
      });

      // Run intersection observer as a macrotask, makes it wait for the next tick
      // before start observing the element to make sure the observer will register the element
      setTimeout(() => {
        if (element) {
          intersectionObserver.observe(element);
        }
      });

      activeObserver.current = intersectionObserver;

      return intersectionObserver;
    },
    [enabled],
  );

  const disconnect = useCallback(() => {
    if (activeObserver.current) {
      activeObserver.current.disconnect();
      activeObserver.current = null;
    }
  }, []);

  const setElementRef = useCallback(
    (ref: Element | null) => {
      // Attempt to disconnect the previous observer
      disconnect();

      // Start a new observer
      startObserver(ref);
    },
    [disconnect, startObserver],
  );

  // Disconnect the observer when the component is unmounted
  useEffect(() => disconnect, [disconnect]);

  // If the feature flag is flipped, append deferred value to the result
  if (shouldDeferLineChartRendering()) {
    // We can safely disable the eslint rule here because flag evaluation is stable
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const [isInViewportDeferred] = useDebounce(isInViewport, deferTimeoutMs);
    return { isInViewport, elementRef: setElementRef, isInViewportDeferred };
  }

  // When the flag is disabled, deferred result is the same as the regular one
  return { isInViewport, elementRef: setElementRef, isInViewportDeferred: isInViewport };
};

import { useEffect, useRef, useState } from 'react';

// Define a module-global observer and a WeakMap on elements to hold callbacks
let sharedObserver: IntersectionObserver | null = null;
const entryCallbackMap = new WeakMap();

const ensureSharedObserverExists = () => {
  if (!sharedObserver) {
    sharedObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const entryCallback = entryCallbackMap.get(entry.target);
        entryCallback?.(entry.isIntersecting);
      }
    });
  }
};

function observeElement(element: Element, callback: (isIntersecting: boolean) => void) {
  ensureSharedObserverExists();
  entryCallbackMap.set(element, callback);
  sharedObserver?.observe(element);

  return () => {
    if (element) {
      sharedObserver?.unobserve(element);
      entryCallbackMap.delete(element);
    }
  };
}

/**
 * Checks if the element is currently visible within the viewport using IntersectionObserver.
 * If "enabled" is set to false, the returned value will always be true.
 */
export const useIsInViewport = <T extends Element>({ enabled = true }: { enabled?: boolean } = {}) => {
  const [element, setElementRef] = useState<T | null>(null);
  const [isInViewport, setIsInViewport] = useState(!enabled);

  useEffect(() => {
    // If already viewed or element is not available, do nothing
    if (!element) {
      return;
    }

    // If IntersectionObserver is not available, assume that the element is visible
    if (!window.IntersectionObserver || !enabled) {
      setIsInViewport(true);
      return;
    }

    return observeElement(element, setIsInViewport);
  }, [enabled, element]);

  // When the flag is disabled, deferred result is the same as the regular one
  return { isInViewport, setElementRef };
};

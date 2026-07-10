import { useEffect, useRef, useState } from 'react';

// Define module-global observers keyed by rootMargin and a WeakMap on elements to hold callbacks
const sharedObservers: Record<string, IntersectionObserver> = {};
const entryCallbackMap = new WeakMap();

const getObserverKey = (rootMargin?: string) => rootMargin ?? 'default';

const ensureSharedObserverExists = (rootMargin?: string) => {
  const observerKey = getObserverKey(rootMargin);
  if (!sharedObservers[observerKey]) {
    sharedObservers[observerKey] = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const entryCallback = entryCallbackMap.get(entry.target);
          entryCallback?.(entry.isIntersecting);
        }
      },
      { rootMargin },
    );
  }
};

function observeElement(element: Element, callback: (isIntersecting: boolean) => void, rootMargin?: string) {
  ensureSharedObserverExists(rootMargin);
  const observerKey = getObserverKey(rootMargin);
  const observer = sharedObservers[observerKey];
  entryCallbackMap.set(element, callback);
  observer?.observe(element);

  return () => {
    if (element) {
      observer?.unobserve(element);
      entryCallbackMap.delete(element);
    }
  };
}

/**
 * Checks if the element is currently visible within the viewport using IntersectionObserver.
 * If "enabled" is set to false, the returned value will always be true.
 */
export const useIsInViewport = <T extends Element>({
  enabled = true,
  rootMargin,
}: { enabled?: boolean; rootMargin?: string } = {}) => {
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

    return observeElement(element, setIsInViewport, rootMargin);
  }, [enabled, element, rootMargin]);

  // When the flag is disabled, deferred result is the same as the regular one
  return { isInViewport, setElementRef };
};

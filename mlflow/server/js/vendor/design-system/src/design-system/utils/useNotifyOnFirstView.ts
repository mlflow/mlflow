import { useEffect, useRef } from 'react';

// Define a module-global observer and a WeakMap on elements to hold callbacks
let sharedObserver: IntersectionObserver | null = null;
const entryCallbackMap = new WeakMap();

const ensureSharedObserverExists = () => {
  if (!sharedObserver) {
    sharedObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const entryCallback = entryCallbackMap.get(entry.target);
          if (entryCallback) {
            entryCallback();
            entryCallbackMap.delete(entry.target);
          }
        }
      });
    });
  }
};

function observeElement(element: Element, callback: () => void) {
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

interface useNotifyOnFirstViewProps {
  onView: () => void;
  resetKey?: unknown;
}

/**
 * Checks if the element was viewed and calls the onView callback.
 * NOTE: This hook only triggers the onView callback once for the element.
 * @param onView - callback to be called when the element is viewed
 * @param resetKey - optional key that resets the observer when changed
 * @typeParam T - extends Element to specify the type of element being observed
 */
export const useNotifyOnFirstView = <T extends Element>({ onView, resetKey }: useNotifyOnFirstViewProps) => {
  const isViewedRef = useRef(false);
  const elementRef = useRef<T | null>(null);
  const previousResetKey = useRef(resetKey);

  // Only reset isViewedRef when resetKey changes
  useEffect(() => {
    if (previousResetKey.current !== resetKey) {
      isViewedRef.current = false;
      previousResetKey.current = resetKey;
    }
  }, [resetKey]);

  useEffect(() => {
    const element = elementRef.current;
    // If already viewed or element is not available, do nothing
    if (!element || isViewedRef.current) {
      return;
    }

    const callback = () => {
      isViewedRef.current = true;
      onView();
    };

    // If IntersectionObserver is not available, assume that the element is visible
    if (!window.IntersectionObserver) {
      callback();
      return;
    }

    return observeElement(element, callback);
  }, [onView]);

  return { elementRef };
};

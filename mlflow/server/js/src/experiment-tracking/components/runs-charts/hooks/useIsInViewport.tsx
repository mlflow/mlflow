import { useEffect, useRef, useState } from 'react';

/**
 * Checks if the element is currently visible within the viewport using IntersectionObserver.
 * If "enabled" is set to false, the returned value will always be true.
 */
export const useIsInViewport = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const internalElementRef = useRef<Element | null>();
  const saveElementRef = (ref: Element | null) => {
    internalElementRef.current = ref;
  };
  const [isInViewport, setIsInViewport] = useState(!enabled);
  // Let's use IntersectionObserver to determine if the element is displayed within the viewport
  useEffect(() => {
    // If "enabled" is set to false or IntersectionObserver is not available, assume that the element is visible
    if (!enabled || !internalElementRef.current || !window.IntersectionObserver) {
      setIsInViewport(true);
      return () => {};
    }

    // Set the state flag only if element is in viewport
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      setIsInViewport(entry.isIntersecting);
    });

    // Run intersection observer as a macrotask, makes it wait for the next tick
    // before start observing the element to make sure the observer will register the element
    setTimeout(() => {
      if (internalElementRef.current) {
        intersectionObserver.observe(internalElementRef.current);
      }
    });

    return () => intersectionObserver.disconnect();
  }, [enabled]);

  return { isInViewport, elementRef: saveElementRef };
};

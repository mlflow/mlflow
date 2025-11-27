import { useCallback, useRef } from 'react';

// If we're within threshold px of the bottom, auto-scroll
const THRESHOLD_PX = 32;

/**
 * Custom hook to manage auto-scrolling to the bottom of a container
 * when new content is added, if the user is already near the bottom.
 */
export const useScrollToBottom = () => {
  const elementRef = useRef<HTMLDivElement>(null);
  const scrollDistanceToBottom = useRef(0);

  const handleScroll = useCallback(() => {
    if (elementRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = elementRef.current;
      // Remember scroll distance to the bottom
      scrollDistanceToBottom.current = scrollHeight - (scrollTop + clientHeight);
    }
  }, []);

  const tryScrollToBottom = useCallback(() => {
    if (elementRef.current && scrollDistanceToBottom.current < THRESHOLD_PX) {
      elementRef.current.scrollTop = elementRef.current.scrollHeight;
    }
  }, []);

  return { elementRef, handleScroll, tryScrollToBottom };
};

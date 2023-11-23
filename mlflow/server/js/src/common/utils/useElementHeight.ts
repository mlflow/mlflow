import { useEffect, useState } from 'react';

/**
 * Hook that returns dynamically updated changing element height. Usage example:
 * ```ts
 * const { elementHeight, observeHeight } = useElementHeight();
 * // ...
 * return <div ref={observeHeight}>Element height: {elementHeight}px</div>
 * ```
 */
export const useElementHeight = () => {
  const [hideableElementsContainer, setHideableElementsContainer] = useState<HTMLElement | null>(
    null,
  );

  const [elementHeight, setElementHeight] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (!hideableElementsContainer || !window.ResizeObserver) {
      return undefined;
    }
    const resizeObserver = new ResizeObserver(([entry]) => {
      if (entry.target.scrollHeight) {
        setElementHeight(entry.target.scrollHeight);
      }
    });
    resizeObserver.observe(hideableElementsContainer);
    return () => resizeObserver.disconnect();
  }, [hideableElementsContainer]);

  return { elementHeight, observeHeight: setHideableElementsContainer };
};

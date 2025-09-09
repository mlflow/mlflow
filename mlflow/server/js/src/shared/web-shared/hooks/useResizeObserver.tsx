import { type RefObject, useRef, useCallback, useEffect, useState } from 'react';

type ResizeObserverOptions<ElementType extends Element> = {
  /**
   * The element to watch for size changes. Can either pass a ref object or a function that when called will return the element to be watched
   */
  ref: RefObject<ElementType | null> | (() => ElementType | null);
  /**
   * Optionally debounces state updates, to prevent rerendering on every single resize
   */
  debounceTimeMs?: number;
};

export function useResizeObserver<ElementType extends Element>({
  ref: rootRef,
  debounceTimeMs,
}: ResizeObserverOptions<ElementType>): { width: number; height: number } | null {
  const prevSize = useRef<{ width: number; height: number }>({ width: -1, height: -1 });
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  const onResize = useCallback(
    (entries: ResizeObserverEntry[]) => {
      if (entries.length === 0) {
        return;
      }

      const rect = entries[0].contentRect;
      if (prevSize.current.width === -1) {
        // First update, just set size immediately
        prevSize.current = { width: rect.width, height: rect.height };
        setSize(prevSize.current);
        return;
      }

      if (rect.width !== prevSize.current.width || rect.height !== prevSize.current.height) {
        prevSize.current.width = rect.width;
        prevSize.current.height = rect.height;

        if (!debounceTimeMs) {
          setSize({ ...prevSize.current });
          return;
        }
        clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => {
          setSize({ ...prevSize.current });
        }, debounceTimeMs);
      }
    },
    [debounceTimeMs],
  );

  const observerRef = useRef<ResizeObserver>();
  if (!observerRef.current) {
    observerRef.current = new ResizeObserver(onResize);
  }

  useEffect(() => {
    const rootElement = typeof rootRef === 'function' ? rootRef() : rootRef.current;
    if (rootElement) {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
      const observer = observerRef.current!;
      observer.observe(rootElement);
      return () => observer.unobserve(rootElement);
    }
    return;
  });

  return size;
}

import type { MutableRefObject } from 'react';
import { useCallback, useLayoutEffect, useRef, useState } from 'react';

export function useResizableMaxWidth(minWidth: number) {
  const ref: MutableRefObject<HTMLDivElement | null> = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number | undefined>(undefined);

  const updateWidth = useCallback(() => {
    if (ref.current) {
      setContainerWidth(ref.current.clientWidth);
    }
  }, []);

  useLayoutEffect(() => {
    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, [updateWidth]);

  const refCallback = useCallback(
    (node: HTMLDivElement) => {
      ref.current = node;
      updateWidth();
    },
    [updateWidth],
  );

  const resizableMaxWidth = containerWidth === undefined ? undefined : containerWidth - minWidth;
  return { resizableMaxWidth, ref: refCallback };
}

import type { RefObject } from 'react';
import { useState, useEffect } from 'react';

interface SizeMap {
  [key: string]: number;
}

function useResponsiveContainer(ref: RefObject<HTMLElement>, sizeMap: SizeMap): string | null {
  const [matchedSize, setMatchedSize] = useState<string | null>(null);

  useEffect(() => {
    if (ref.current && sizeMap) {
      const handleResize = () => {
        if (!ref.current) {
          return;
        }
        const elementWidth = ref.current.offsetWidth;
        const matchedKey = Object.keys(sizeMap)
          .filter((key) => sizeMap[key] >= elementWidth)
          .sort((a, b) => sizeMap[a] - sizeMap[b])[0];

        setMatchedSize(matchedKey);
      };

      handleResize();

      const resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(ref.current);

      return () => resizeObserver.disconnect();
    }
    return undefined;
  }, [ref, sizeMap]);

  return matchedSize;
}

export default useResponsiveContainer;

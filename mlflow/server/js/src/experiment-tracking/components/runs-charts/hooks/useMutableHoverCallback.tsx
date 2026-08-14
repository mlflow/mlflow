import type { PlotMouseEvent } from 'plotly.js';
import { useEffect, useRef } from 'react';

/**
 * Unfortunately plotly.js memorizes first onHover callback given on initial render,
 * so in order to achieve updated behavior we need to wrap each onHover callback with an
 * immutable callback that will call mutable implementation.
 */
export const useMutableChartHoverCallback = <T extends (event: Readonly<PlotMouseEvent>) => void>(callback: T) => {
  const mutableRef = useRef<T>(callback);

  useEffect(() => {
    mutableRef.current = callback;
  }, [callback]);

  return (event: Readonly<PlotMouseEvent>) => {
    mutableRef.current(event);
  };
};

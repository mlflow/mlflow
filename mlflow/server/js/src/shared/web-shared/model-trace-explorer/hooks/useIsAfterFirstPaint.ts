import { useEffect, useState } from 'react';
import { unstable_batchedUpdates } from 'react-dom';

let pendingCallbacks: Set<() => void> | undefined;

/** Uses the initial defer value only; deferred callers become ready once after paint and remain ready. */
export const useIsAfterFirstPaint = (defer: boolean): boolean => {
  const [ready, setReady] = useState(!defer);
  useEffect(() => {
    if (ready) return;
    if (!pendingCallbacks) {
      const callbacks = new Set<() => void>();
      pendingCallbacks = callbacks;
      requestAnimationFrame(() => {
        // A timer from the frame runs after the browser has had a chance to paint.
        setTimeout(() => {
          pendingCallbacks = undefined;
          unstable_batchedUpdates(() => callbacks.forEach((callback) => callback()));
        }, 0);
      });
    }
    const callbacks = pendingCallbacks;
    const onPaint = () => setReady(true);
    callbacks.add(onPaint);
    return () => {
      callbacks.delete(onPaint);
    };
  }, [ready]);
  return ready;
};

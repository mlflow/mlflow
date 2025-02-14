import { useRef } from 'react';

// Use Symbol to correctly handled null/undefined as stable values
const NOT_INITIALIZED = Symbol('NOT_INITIALIZED');

/**
 * Use to keep a stable value on the first mount of a component
 * that cannot change.
 */
function useStable<T>(valueCallback: () => T): T {
  const ref = useRef<T | symbol>(NOT_INITIALIZED);
  if (ref.current === NOT_INITIALIZED) {
    const val = valueCallback();
    ref.current = val;
    return val;
  }
  return ref.current as T;
}

let sequentialCounter = 0;

export function useStableUid() {
  return useStable(() => sequentialCounter++);
}

import { isEqual } from 'lodash';
import { useRef } from 'react';

/**
 * Utility hook that memoizes value based on deep comparison.
 * Dedicated to a few limited use cases where deep comparison is still cheaper than resulting re-renders.
 */
export const useMemoDeep = <T>(factory: () => T, deps: unknown[]): T => {
  const ref = useRef<{ deps: unknown[]; value: T }>();

  if (!ref.current || !isEqual(deps, ref.current.deps)) {
    ref.current = { deps, value: factory() };
  }

  return ref.current.value;
};

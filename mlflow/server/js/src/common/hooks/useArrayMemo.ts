import { useRef } from 'react';

/**
 * A custom hook that memoizes an array based on the reference of its elements, not the array itself.
 */
export function useArrayMemo<T>(array: T[]) {
  // This holds reference to previous value
  const ref = useRef<T[]>();
  // Check if each element of the old and new array match
  const areArraysConsideredTheSame =
    ref.current && array.length === ref.current.length
      ? array.every((element, i) => {
          return element === ref.current?.[i];
        })
      : // Initially there's no old array defined/stored, so set to false
        false;

  if (!areArraysConsideredTheSame) {
    ref.current = array;
  }

  return areArraysConsideredTheSame && ref.current ? ref.current : array;
}

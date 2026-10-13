import { useCallback } from 'react';
import { useSearchParamSelector } from './useSearchParamSelector';

interface UseNumberSearchParamOptions {
  /** URL search-param key. */
  key: string;
  /** Returned when the param is missing or unparseable. */
  defaultValue: number;
  /** Lowest legal value. Anything lower is clamped up. */
  min?: number;
}

const parseNumber = (raw: string | null, defaultValue: number, min: number): number => {
  if (raw === null) return defaultValue;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return defaultValue;
  return parsed < min ? min : parsed;
};

/**
 * Read/write an integer URL search param with a default and minimum clamp.
 * Writing the default value removes the param so URLs stay tidy.
 */
export const useNumberSearchParam = ({
  key,
  defaultValue,
  min = 1,
}: UseNumberSearchParamOptions): [number, (next: number) => void] => {
  const [value, setSearchParams] = useSearchParamSelector((params) => parseNumber(params.get(key), defaultValue, min));

  const setValue = useCallback(
    (next: number) => {
      setSearchParams((params) => {
        if (next === defaultValue) {
          params.delete(key);
        } else {
          params.set(key, String(next));
        }
        return params;
      });
    },
    [setSearchParams, key, defaultValue],
  );

  return [value, setValue];
};

import { useState, useCallback } from 'react';

import { buildStorageKey, type UseLocalStorageParams } from './useLocalStorage';

/** Params for {@link useSessionStorage} (identical in shape to {@link UseLocalStorageParams}). */
export type UseSessionStorageParams<T> = UseLocalStorageParams<T>;

/**
 * Retrieves a value from sessionStorage based on the provided key.
 * If the value is not found or cannot be parsed as JSON, the initial value is returned.
 */
export const getSessionStorageItem = <T>(key: string, version: number, initialValue: T): T => {
  const fullKey = buildStorageKey(key, version);
  try {
    // eslint-disable-next-line @databricks/no-direct-storage -- go/no-direct-storage
    const item = window.sessionStorage.getItem(fullKey);
    return item ? JSON.parse(item) : initialValue;
  } catch (error) {
    return initialValue;
  }
};

/**
 * Sets a value in sessionStorage for the provided key.
 */
export const setSessionStorageItem = <T>(
  key: string,
  version: number,
  value: T,
  onFailure?: (error: unknown) => void,
): void => {
  const fullKey = buildStorageKey(key, version);
  const valueToSet = JSON.stringify(value);
  try {
    // eslint-disable-next-line @databricks/no-direct-storage -- go/no-direct-storage
    window.sessionStorage.setItem(fullKey, valueToSet);
  } catch (error) {
    onFailure?.(error);
  }
};

/**
 * Works like useState() but backed by sessionStorage, so the value survives reloads
 * within the same tab while staying isolated per tab. Use this instead of
 * {@link useLocalStorage} when state must NOT be shared across tabs (e.g. a per-tab
 * conversation that two tabs would otherwise overwrite for each other).
 */
export function useSessionStorage<T>({ key, version, initialValue, onFailure }: UseSessionStorageParams<T>) {
  const [storedValue, setStoredValue] = useState<T>(() => getSessionStorageItem<T>(key, version, initialValue));

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      setStoredValue((prev) => {
        const newValue = value instanceof Function ? value(prev) : value;
        setSessionStorageItem<T>(key, version, newValue, onFailure);
        return newValue;
      });
    },
    [key, version, onFailure],
  );

  return [storedValue, setValue] as const;
}

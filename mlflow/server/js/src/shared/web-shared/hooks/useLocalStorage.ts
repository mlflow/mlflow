import { useState, useCallback } from 'react';

/** Params that describe a local storage key for use with {@link useLocalStorage}. */
export interface UseLocalStorageParams<T> {
  key: string;
  version: number;
  initialValue: T;
  onFailure?: (error: unknown) => void;
}

export const buildStorageKey = (baseKey: string, version: number): string => {
  const storageKey = `${baseKey}_v${version}`;
  return storageKey;
};

/**
 * Builds a storage key based on the provided {@link UseLocalStorageParams}.
 *
 * For details see {@link buildStorageKey}.
 */
export function buildStorageKeyByParams({ key, version }: Omit<UseLocalStorageParams<any>, 'initialValue'>): string {
  return buildStorageKey(key, version);
}

/**
 * Retrieves a value from the localStorage based on the provided key.
 * If the value is not found or cannot be parsed as JSON, the initial value is returned.
 *
 * @param {string} key - The key to retrieve the value from localStorage.
 * @param {number} version - The version of the storage key.
 * @param {*} initialValue - The initial value to return if the value is not found or cannot be parsed.
 * @returns {*} - The retrieved value or the initial value if not found or cannot be parsed.
 *
 * @example
 * // Example 1: Retrieving a stored value
 * const storedValue = getLocalStorageItem("username", 0.1, true, "Guest");
 * // Returns the value stored in localStorage with the key "username",
 * // or the initial value "Guest" if the key is not found or cannot be parsed.
 *
 * // Example 2: Retrieving a non-existent value
 * const nonExistentValue = getLocalStorageItem("email", 0.1, true, null);
 * // Returns null as the initial value is used since the key "email" is not found.
 */
export const getLocalStorageItem = <T>(key: string, version: number, initialValue: T): T => {
  const fullKey = buildStorageKey(key, version);
  try {
    const item = window.localStorage.getItem(fullKey);
    return item ? JSON.parse(item) : initialValue;
  } catch (error) {
    return initialValue;
  }
};

/**
 * Retrieves a value from local storage for the provided {@link UseLocalStorageParams}.
 *
 * For details see {@link getLocalStorageItem}.
 */
export function getLocalStorageItemByParams<T>({ key, version, initialValue }: UseLocalStorageParams<T>): T {
  return getLocalStorageItem(key, version, initialValue);
}

/**
 * Sets a value in the localStorage for the provided key.
 *
 * @param key - The key to set the value in localStorage.
 * @param {number} version - The version of the storage key.
 * @param {*} value - The value to be stored in localStorage.
 * @param {function} onFailure - The function to be called if an error occurs while setting the value.
 * @returns {void}
 *
 * @example
 * // Example 1: Storing a string value
 * setLocalStorageItem("username", 0.1, true, "John");
 *
 * // Example 2: Storing an object
 * const user = { name: "Alice", age: 25 };
 * setLocalStorageItem("user", 0.1, true, user);
 */
export const setLocalStorageItem = <T>(
  key: string,
  version: number,
  value: T,
  onFailure?: (error: unknown) => void,
): void => {
  const fullKey = buildStorageKey(key, version);
  const valueToSet = JSON.stringify(value);

  try {
    localStorage.setItem(fullKey, valueToSet);
  } catch (error) {
    onFailure?.(error);
  }
};

/**
 * Sets a value in local storage for the provided {@link UseLocalStorageParams}.
 *
 * For details see {@link setLocalStorageItem}.
 */
export function setLocalStorageItemByParams<T>(
  { key, version }: Omit<UseLocalStorageParams<T>, 'initialValue'>,
  value: T,
  onFailure?: (error: unknown) => void,
) {
  setLocalStorageItem(key, version, value, onFailure);
}

/**
 * It works similar to useState() but uses local storage for backing the data
 * WARNING: Do not reuse the same key in more than one place
 * Think about what will happen if two tabs are opened and they overwrite each other's value
 */
export function useLocalStorage<T>({ key, version, initialValue, onFailure }: UseLocalStorageParams<T>) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    return getLocalStorageItem<T>(key, version, initialValue);
  });

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      setStoredValue((prev) => {
        const newValue = value instanceof Function ? value(prev) : value;
        setLocalStorageItem<T>(key, version, newValue, onFailure);
        return newValue;
      });
    },
    [key, version, onFailure],
  );

  return [storedValue, setValue] as const;
}

import { describe, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useSessionStorage, getSessionStorageItem, setSessionStorageItem } from './useSessionStorage';

describe('session storage helpers', () => {
  test('getSessionStorageItem returns the initial value when nothing is stored', () => {
    expect(getSessionStorageItem('absent-key', 1, 'fallback')).toBe('fallback');
  });

  test('setSessionStorageItem then getSessionStorageItem round-trips a value', () => {
    setSessionStorageItem('round-trip', 1, { a: 1, b: 'two' });
    expect(getSessionStorageItem('round-trip', 1, null)).toEqual({ a: 1, b: 'two' });
  });

  test('values are namespaced by version', () => {
    setSessionStorageItem('versioned', 1, 'v1-value');
    expect(getSessionStorageItem('versioned', 2, 'v2-default')).toBe('v2-default');
  });
});

describe('useSessionStorage', () => {
  test('seeds from previously stored value', () => {
    setSessionStorageItem('seeded', 1, 'stored');
    const { result } = renderHook(() => useSessionStorage({ key: 'seeded', version: 1, initialValue: 'default' }));
    expect(result.current[0]).toBe('stored');
  });

  test('falls back to initialValue when nothing is stored', () => {
    const { result } = renderHook(() => useSessionStorage({ key: 'fresh', version: 1, initialValue: 'default' }));
    expect(result.current[0]).toBe('default');
  });

  test('setValue updates state and persists to sessionStorage', () => {
    const { result } = renderHook(() => useSessionStorage({ key: 'writes', version: 1, initialValue: 0 }));
    act(() => result.current[1](42));
    expect(result.current[0]).toBe(42);
    expect(getSessionStorageItem('writes', 1, -1)).toBe(42);
  });

  test('setValue supports a functional updater', () => {
    const { result } = renderHook(() => useSessionStorage({ key: 'updater', version: 1, initialValue: 1 }));
    act(() => result.current[1]((prev) => prev + 9));
    expect(result.current[0]).toBe(10);
    expect(getSessionStorageItem('updater', 1, -1)).toBe(10);
  });
});

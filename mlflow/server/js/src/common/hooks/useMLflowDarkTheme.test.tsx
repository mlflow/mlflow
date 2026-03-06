import { describe, beforeEach, test, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useMLflowDarkTheme } from './useMLflowDarkTheme';

// ------- matchMedia mock -------
type MediaQueryListener = (event: { matches: boolean }) => void;

let listeners: MediaQueryListener[] = [];
let systemDarkMode = false;

function createMatchMediaMock() {
  listeners = [];
  return (query: string) => ({
    matches: query === '(prefers-color-scheme: dark)' ? systemDarkMode : false,
    media: query,
    addEventListener: (_event: string, cb: MediaQueryListener) => {
      listeners.push(cb);
    },
    removeEventListener: (_event: string, cb: MediaQueryListener) => {
      listeners = listeners.filter((l) => l !== cb);
    },
    // Unused stubs required by the MediaQueryList interface.
    addListener: () => {},
    removeListener: () => {},
    onchange: null,
    dispatchEvent: () => false,
  });
}

/** Simulate the OS switching colour scheme. */
function simulateSystemThemeChange(dark: boolean) {
  systemDarkMode = dark;
  listeners.forEach((cb) => cb({ matches: dark }));
}

// ------- helpers -------
beforeEach(() => {
  localStorage.clear();
  document.body.classList.remove('dark-mode');
  systemDarkMode = false;
  Object.defineProperty(window, 'matchMedia', { writable: true, value: createMatchMediaMock() });
});

describe('useMLflowDarkTheme', () => {
  // ---- default / system behaviour ----

  test('defaults to system preference (light)', () => {
    systemDarkMode = false;
    Object.defineProperty(window, 'matchMedia', { writable: true, value: createMatchMediaMock() });

    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(result.current.isDarkTheme).toBe(false);
    expect(result.current.themePreference).toBe('system');
  });

  test('defaults to system preference (dark)', () => {
    systemDarkMode = true;
    Object.defineProperty(window, 'matchMedia', { writable: true, value: createMatchMediaMock() });

    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(result.current.isDarkTheme).toBe(true);
    expect(result.current.themePreference).toBe('system');
  });

  test('follows system changes in system mode', () => {
    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(result.current.isDarkTheme).toBe(false);

    act(() => simulateSystemThemeChange(true));
    expect(result.current.isDarkTheme).toBe(true);

    act(() => simulateSystemThemeChange(false));
    expect(result.current.isDarkTheme).toBe(false);
  });

  // ---- manual override ----

  test('manual toggle overrides system preference', () => {
    const { result } = renderHook(() => useMLflowDarkTheme());

    act(() => result.current.setIsDarkTheme(true));
    expect(result.current.isDarkTheme).toBe(true);
    expect(result.current.themePreference).toBe('dark');

    // System change should now be ignored.
    act(() => simulateSystemThemeChange(false));
    expect(result.current.isDarkTheme).toBe(true);
  });

  test('manual toggle persists to localStorage', () => {
    const { result } = renderHook(() => useMLflowDarkTheme());

    act(() => result.current.setIsDarkTheme(true));
    expect(localStorage.getItem('_mlflow_dark_mode_preference')).toBe('dark');
    expect(localStorage.getItem('_mlflow_dark_mode_toggle_enabled')).toBe('true');
    expect(localStorage.getItem('databricks-dark-mode-pref')).toBe('dark');
  });

  // ---- setUseSystemTheme ----

  test('setUseSystemTheme re-enables system tracking', () => {
    systemDarkMode = true;
    Object.defineProperty(window, 'matchMedia', { writable: true, value: createMatchMediaMock() });

    const { result } = renderHook(() => useMLflowDarkTheme());

    // Override to light.
    act(() => result.current.setIsDarkTheme(false));
    expect(result.current.isDarkTheme).toBe(false);

    // Reset to system — should pick up the current system value (dark).
    act(() => result.current.setUseSystemTheme());
    expect(result.current.themePreference).toBe('system');
    expect(result.current.isDarkTheme).toBe(true);
  });

  // ---- body class ----

  test('updates body class when theme changes', () => {
    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(document.body.classList.contains('dark-mode')).toBe(false);

    act(() => result.current.setIsDarkTheme(true));
    expect(document.body.classList.contains('dark-mode')).toBe(true);

    act(() => result.current.setIsDarkTheme(false));
    expect(document.body.classList.contains('dark-mode')).toBe(false);
  });

  // ---- migration from legacy key ----

  test('migrates legacy "true" preference on first load', () => {
    localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'true');
    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(result.current.isDarkTheme).toBe(true);
    expect(result.current.themePreference).toBe('dark');
  });

  test('migrates legacy "false" preference on first load', () => {
    localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'false');
    const { result } = renderHook(() => useMLflowDarkTheme());
    expect(result.current.isDarkTheme).toBe(false);
    expect(result.current.themePreference).toBe('light');
  });

  test('new preference key takes precedence over legacy key', () => {
    localStorage.setItem('_mlflow_dark_mode_preference', 'system');
    localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'true');
    const { result } = renderHook(() => useMLflowDarkTheme());
    // 'system' wins; legacy key is ignored.
    expect(result.current.themePreference).toBe('system');
  });

  // ---- setter stability ----

  test('setIsDarkTheme identity is stable across renders', () => {
    const { result, rerender } = renderHook(() => useMLflowDarkTheme());
    const first = result.current.setIsDarkTheme;
    rerender();
    expect(result.current.setIsDarkTheme).toBe(first);
  });

  test('setUseSystemTheme identity is stable across renders', () => {
    const { result, rerender } = renderHook(() => useMLflowDarkTheme());
    const first = result.current.setUseSystemTheme;
    rerender();
    expect(result.current.setUseSystemTheme).toBe(first);
  });
});

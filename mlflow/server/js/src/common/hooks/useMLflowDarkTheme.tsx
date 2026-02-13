import { Global } from '@emotion/react';
import { useCallback, useEffect, useMemo, useState } from 'react';

// Compatibility keys consumed by the bundled JS / design-system layer.
const databricksDarkModePrefLocalStorageKey = 'databricks-dark-mode-pref';
const darkModePrefLocalStorageKey = '_mlflow_dark_mode_toggle_enabled';
// Preference key introduced with the system-theme feature.
const darkModePreferenceLocalStorageKey = '_mlflow_dark_mode_preference';
const darkModeBodyClassName = 'dark-mode';

export type DarkModePreference = 'system' | 'dark' | 'light';

export interface MLflowDarkThemeResult {
  isDarkTheme: boolean;
  setIsDarkTheme: (value: boolean) => void;
  ThemeGlobalStyles: React.ComponentType<React.PropsWithChildren<unknown>>;
  setUseSystemTheme: () => void;
  themePreference: DarkModePreference;
}

// CSS attributes to be applied when dark mode is enabled. Affects inputs and other form elements.
const darkModeCSSStyles = { body: { [`&.${darkModeBodyClassName}`]: { colorScheme: 'dark' } } };
// This component is used to set the global CSS.
const DarkModeStylesComponent = () => <Global styles={darkModeCSSStyles} />;

/**
 * Reads the persisted preference, falling back to 'system' for new users.
 *
 * Migration: users who previously had `_mlflow_dark_mode_toggle_enabled` set
 * (before the preference key existed) get their explicit choice migrated once;
 * after that the new key is authoritative.
 */
function readInitialPreference(): DarkModePreference {
  const stored = localStorage.getItem(darkModePreferenceLocalStorageKey);
  if (stored === 'dark' || stored === 'light' || stored === 'system') {
    return stored;
  }

  // One-time migration from the legacy boolean key.
  const legacy = localStorage.getItem(darkModePrefLocalStorageKey);
  if (legacy === 'true') {
    return 'dark';
  }
  if (legacy === 'false') {
    return 'light';
  }

  return 'system';
}

/**
 * This hook is used to toggle the dark mode for the entire app.
 * Used in open source MLflow.
 *
 * Behaviour:
 * - First visit (no stored preference): follows the OS / browser colour-scheme.
 * - Manual toggle: persists 'dark' | 'light' and stops auto-switching.
 * - "Use system" action: resets to 'system' and resumes auto-switching.
 */
export const useMLflowDarkTheme = (): MLflowDarkThemeResult => {
  const darkModeMediaQuery = useMemo(() => window.matchMedia('(prefers-color-scheme: dark)'), []);

  const [systemPrefersDark, setSystemPrefersDark] = useState(() => darkModeMediaQuery.matches);
  const [themePreference, setThemePreference] = useState<DarkModePreference>(readInitialPreference);

  const isDarkTheme = themePreference === 'system' ? systemPrefersDark : themePreference === 'dark';

  // Track OS-level colour-scheme changes.
  useEffect(() => {
    const handleChange = (event: MediaQueryListEvent) => {
      setSystemPrefersDark(event.matches);
    };
    darkModeMediaQuery.addEventListener('change', handleChange);
    return () => darkModeMediaQuery.removeEventListener('change', handleChange);
  }, [darkModeMediaQuery]);

  // Explicit dark/light override — no stale-closure risk since it never
  // needs to read the current derived value.
  const setIsDarkTheme = useCallback((value: boolean) => {
    setThemePreference(value ? 'dark' : 'light');
  }, []);

  const setUseSystemTheme = useCallback(() => {
    setThemePreference('system');
  }, []);

  // Persist preference + update compatibility keys in a single effect.
  useEffect(() => {
    localStorage.setItem(darkModePreferenceLocalStorageKey, themePreference);
    document.body.classList.toggle(darkModeBodyClassName, isDarkTheme);
    localStorage.setItem(darkModePrefLocalStorageKey, isDarkTheme ? 'true' : 'false');
    localStorage.setItem(databricksDarkModePrefLocalStorageKey, isDarkTheme ? 'dark' : 'light');
  }, [themePreference, isDarkTheme]);

  return { isDarkTheme, setIsDarkTheme, ThemeGlobalStyles: DarkModeStylesComponent, setUseSystemTheme, themePreference };
};

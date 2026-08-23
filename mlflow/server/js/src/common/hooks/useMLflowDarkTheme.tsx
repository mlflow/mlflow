import { Global } from '@emotion/react';
import { useEffect, useState } from 'react';
import { useMediaQuery } from '@databricks/web-shared/hooks';

export type MLflowThemePreference = 'system' | 'light' | 'dark';

// bundled JS needs to read this key in order to enable dark mode
const databricksDarkModePrefLocalStorageKey = 'databricks-dark-mode-pref';
const themePrefLocalStorageKey = '_mlflow_dark_mode_toggle_enabled';
const darkModeBodyClassName = 'dark-mode';

const parseThemePreference = (value: string | null): MLflowThemePreference => {
  // Only honor explicit light/dark choices. Older builds persisted a boolean
  // here on every page load, even without a manual toggle, so legacy values
  // fall back to "system" rather than being treated as a deliberate choice.
  if (value === 'dark') {
    return 'dark';
  }
  if (value === 'light') {
    return 'light';
  }
  return 'system';
};

// CSS attributes to be applied when dark mode is enabled. Affects inputs and other form elements.
const darkModeCSSStyles = { body: { [`&.${darkModeBodyClassName}`]: { colorScheme: 'dark' } } };
// This component is used to set the global CSS.
const DarkModeStylesComponent = () => <Global styles={darkModeCSSStyles} />;

/**
 * This hook manages the color mode for the entire app.
 * Supports three preferences: "system" (default) follows the OS color scheme and
 * updates live when it changes, while "light" and "dark" override it explicitly.
 * Used in open source MLflow.
 * Returns a boolean value with the effective state, the preference value with its setter,
 * and a component to be rendered in the root of the app.
 */
export const useMLflowDarkTheme = (): [
  boolean,
  MLflowThemePreference,
  (themePreference: MLflowThemePreference) => void,
  React.ComponentType<React.PropsWithChildren<unknown>>,
] => {
  const [themePreference, setThemePreference] = useState<MLflowThemePreference>(() => {
    // eslint-disable-next-line @databricks/no-direct-storage -- go/no-direct-storage
    return parseThemePreference(localStorage.getItem(themePrefLocalStorageKey));
  });

  // Re-renders whenever the OS color-scheme preference changes, making
  // "system" mode follow it live.
  const systemPrefersDark = useMediaQuery('(prefers-color-scheme: dark)');
  const isDarkTheme = themePreference === 'system' ? systemPrefersDark : themePreference === 'dark';

  useEffect(() => {
    // Update the theme when the preference or the system scheme changes.
    document.body.classList.toggle(darkModeBodyClassName, isDarkTheme);
    // Persist the user's preference in local storage.
    // eslint-disable-next-line @databricks/no-direct-storage -- go/no-direct-storage
    localStorage.setItem(themePrefLocalStorageKey, themePreference);
    // eslint-disable-next-line @databricks/no-direct-storage -- go/no-direct-storage
    localStorage.setItem(databricksDarkModePrefLocalStorageKey, isDarkTheme ? 'dark' : 'light');
  }, [isDarkTheme, themePreference]);

  return [isDarkTheme, themePreference, setThemePreference, DarkModeStylesComponent];
};

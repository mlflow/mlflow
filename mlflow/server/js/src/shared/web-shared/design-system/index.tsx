import React from 'react';

import { DesignSystemThemeProvider } from '@databricks/design-system';

export type DarkModePref = 'system' | 'dark' | 'light';

const DARK_MODE_PREF_DEFAULT = 'light';

export interface SupportsDuBoisThemesProps {
  isDarkMode?: boolean;
}

export const SupportsDuBoisThemes: React.FC<SupportsDuBoisThemesProps> = ({
  isDarkMode = false,
  children,
}) => {
  // eslint-disable-next-line react/forbid-elements
  return <DesignSystemThemeProvider isDarkMode={isDarkMode}>{children}</DesignSystemThemeProvider>;
};

export function getUserDarkModePref(): DarkModePref {
  return 'system';
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function setUserDarkModePref(value: DarkModePref) {}

// For system-level dark mode preference
const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

export function systemPrefersDark(): boolean {
  return darkModeMediaQuery.matches;
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
export function setDarkModeSupported(value: boolean) {}

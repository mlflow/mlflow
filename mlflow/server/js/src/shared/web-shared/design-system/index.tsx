import React from 'react';

import { DesignSystemThemeProvider } from '@databricks/design-system';

export type DarkModePref = 'system' | 'dark' | 'light';

const DARK_MODE_PREF_DEFAULT = 'light';

export interface SupportsDuBoisThemesProps {
  enabled?: boolean;
}

export const SupportsDuBoisThemes: React.FC<SupportsDuBoisThemesProps> = ({
  enabled = false,
  children,
}) => {
  return <DesignSystemThemeProvider isDarkMode={enabled}>{children}</DesignSystemThemeProvider>;
};

export function getUserDarkModePref(): DarkModePref {
  return 'system';
}

export function setUserDarkModePref(value: DarkModePref) {}

// For system-level dark mode preference
const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

export function systemPrefersDark(): boolean {
  return darkModeMediaQuery.matches;
}

export function setDarkModeSupported(value: boolean) {}

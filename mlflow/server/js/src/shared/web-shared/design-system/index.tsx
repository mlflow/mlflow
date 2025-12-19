import React from 'react';

import { DesignSystemThemeProvider } from '@databricks/design-system';

export type DarkModePref = 'system' | 'dark' | 'light';

export const DARK_MODE_PREF_DEFAULT = 'system';

export const LOCAL_STORAGE_DARK_MODE_PREF_KEY: any = 'dark-mode-pref';

export interface SupportsDuBoisThemesProps {
  disabled?: boolean;
}

export const SupportsDuBoisThemes: React.FC<React.PropsWithChildren<SupportsDuBoisThemesProps>> = ({
  disabled = false,
  children,
}) => {
  // eslint-disable-next-line react/forbid-elements
  return <DesignSystemThemeProvider isDarkMode={false}>{children}</DesignSystemThemeProvider>;
};

export const getIsDarkMode = (prefersDarkMode: DarkModePref): boolean => {
  return prefersDarkMode === 'dark';
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

export function WorkspaceImg(props: React.ImgHTMLAttributes<HTMLImageElement> & { src?: string; alt?: string }) {
  return <img alt="altt" {...props} />;
}

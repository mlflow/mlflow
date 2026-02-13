import React, { createContext, useContext, useMemo } from 'react';
import type { DarkModePreference } from '../hooks/useMLflowDarkTheme';

interface DarkThemeContextType {
  isDarkTheme: boolean;
  setIsDarkTheme: (isDarkTheme: boolean) => void;
  setUseSystemTheme: () => void;
  themePreference: DarkModePreference;
}

const DarkThemeContext = createContext<DarkThemeContextType>({
  isDarkTheme: false,
  setIsDarkTheme: () => {},
  setUseSystemTheme: () => {},
  themePreference: 'system',
});

export const DarkThemeProvider = ({
  children,
  isDarkTheme,
  setIsDarkTheme,
  setUseSystemTheme,
  themePreference,
}: {
  children: React.ReactNode;
  isDarkTheme: boolean;
  setIsDarkTheme: (isDarkTheme: boolean) => void;
  setUseSystemTheme: () => void;
  themePreference: DarkModePreference;
}) => {
  const value = useMemo(
    () => ({ isDarkTheme, setIsDarkTheme, setUseSystemTheme, themePreference }),
    [isDarkTheme, setIsDarkTheme, setUseSystemTheme, themePreference],
  );
  return <DarkThemeContext.Provider value={value}>{children}</DarkThemeContext.Provider>;
};

export const useDarkThemeContext = () => useContext(DarkThemeContext);

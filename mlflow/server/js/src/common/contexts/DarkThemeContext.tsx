import React, { createContext, useContext } from 'react';
import type { MLflowThemePreference } from '../hooks/useMLflowDarkTheme';

interface DarkThemeContextType {
  themePreference: MLflowThemePreference;
  setThemePreference: (themePreference: MLflowThemePreference) => void;
}

const DarkThemeContext = createContext<DarkThemeContextType>({
  themePreference: 'system',
  setThemePreference: () => {},
});

export const DarkThemeProvider = ({
  children,
  themePreference,
  setThemePreference,
}: {
  children: React.ReactNode;
  themePreference: MLflowThemePreference;
  setThemePreference: (themePreference: MLflowThemePreference) => void;
}) => {
  return (
    <DarkThemeContext.Provider value={{ themePreference, setThemePreference }}>{children}</DarkThemeContext.Provider>
  );
};

export const useDarkThemeContext = () => useContext(DarkThemeContext);

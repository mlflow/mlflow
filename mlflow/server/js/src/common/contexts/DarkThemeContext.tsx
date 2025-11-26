import React, { createContext, useContext } from 'react';

interface DarkThemeContextType {
  isDarkTheme: boolean;
  setIsDarkTheme: (isDarkTheme: boolean) => void;
}

const DarkThemeContext = createContext<DarkThemeContextType>({
  isDarkTheme: false,
  setIsDarkTheme: () => {},
});

export const DarkThemeProvider = ({
  children,
  isDarkTheme,
  setIsDarkTheme,
}: {
  children: React.ReactNode;
  isDarkTheme: boolean;
  setIsDarkTheme: (isDarkTheme: boolean) => void;
}) => {
  return <DarkThemeContext.Provider value={{ isDarkTheme, setIsDarkTheme }}>{children}</DarkThemeContext.Provider>;
};

export const useDarkThemeContext = () => useContext(DarkThemeContext);

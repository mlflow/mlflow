import React, { createContext, useContext } from 'react';

interface DarkThemeContextType {
  setIsDarkTheme: (isDarkTheme: boolean) => void;
}

const DarkThemeContext = createContext<DarkThemeContextType>({
  setIsDarkTheme: () => {},
});

export const DarkThemeProvider = ({
  children,
  setIsDarkTheme,
}: {
  children: React.ReactNode;
  setIsDarkTheme: (isDarkTheme: boolean) => void;
}) => {
  return <DarkThemeContext.Provider value={{ setIsDarkTheme }}>{children}</DarkThemeContext.Provider>;
};

export const useDarkThemeContext = () => useContext(DarkThemeContext);

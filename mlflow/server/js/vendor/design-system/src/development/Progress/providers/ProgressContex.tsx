import { createContext } from 'react';

export interface ProgressContextType {
  progress?: number | null;
}

const progressContextDefaults: ProgressContextType = {
  progress: 0,
};

export const ProgressContext = createContext<ProgressContextType>(progressContextDefaults);

export const ProgressContextProvider = ({ children, value }: { children: JSX.Element; value: ProgressContextType }) => {
  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>;
};
